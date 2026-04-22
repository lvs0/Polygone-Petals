//! Polygone-Petals network layer using unified P2P infrastructure.
//!
//! This module provides Petals-specific networking on top of the shared
//! Polygone P2P layer, handling distributed inference requests.

use libp2p::{
    identify, kad, request_response, swarm::NetworkBehaviour, StreamProtocol
};
use polygone::{
    network::{
        P2pNode, P2pConfig, NetworkEvent, PolygoneRequest, PolygoneResponse,
        Capability, GossipMessage, Multiaddr, PeerId,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Sequential inference request (legacy format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub session_id: [u8; 16],
    pub start_layer: u32,
    pub end_layer: u32,
    pub hidden_states_data: Vec<u8>,
    pub dims: Vec<usize>,
}

/// Response signaling successful computation (legacy format).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResponse {
    pub success: bool,
    pub outgoing_data: Option<Vec<u8>>,
}

/// Petals compute network node
pub struct PetalsNetwork {
    p2p_node: P2pNode,
    event_rx: mpsc::Receiver<NetworkEvent>,
    /// Layer range this node can compute
    layer_start: u32,
    layer_end: u32,
    /// Active inference sessions
    active_sessions: HashMap<[u8; 16], SessionInfo>,
}

struct SessionInfo {
    start_time: std::time::Instant,
    peer_id: PeerId,
}

impl PetalsNetwork {
    /// Create a new Petals compute node
    pub async fn new(
        config: P2pConfig,
        layer_start: u32,
        layer_end: u32,
    ) -> anyhow::Result<Self> {
        let (p2p_node, event_rx) = P2pNode::new(config).await?;
        
        Ok(Self {
            p2p_node,
            event_rx,
            layer_start,
            layer_end,
            active_sessions: HashMap::new(),
        })
    }

    /// Get the local PeerId
    pub fn peer_id(&self) -> PeerId {
        self.p2p_node.peer_id()
    }

    /// Start listening and announce capabilities
    pub async fn start(&mut self, bootstrap_addrs: Vec<Multiaddr>) -> anyhow::Result<()> {
        let addrs = self.p2p_node.start_listening().await?;
        info!("Petals node listening on: {:?}", addrs);

        // Subscribe to petals topic
        self.p2p_node.subscribe_topic("polygone-petals")?;

        // Bootstrap
        for addr in bootstrap_addrs {
            info!("Bootstrapping to: {}", addr);
        }
        self.p2p_node.bootstrap().await?;

        // Announce compute capabilities
        self.announce_capabilities().await?;

        Ok(())
    }

    /// Announce compute capabilities to the network
    async fn announce_capabilities(&mut self) -> anyhow::Result<()> {
        let message = GossipMessage::CapabilitiesAnnounce {
            peer_id: self.peer_id().to_bytes(),
            capabilities: vec![Capability::PetalsCompute {
                layer_start: self.layer_start,
                layer_end: self.layer_end,
            }],
            ttl_seconds: 3600,
        };
        
        self.p2p_node.publish_gossip("polygone-petals", message)?;
        info!(
            "Announced compute capability: layers {}-{}",
            self.layer_start, self.layer_end
        );
        
        Ok(())
    }

    /// Request inference from a peer
    pub async fn request_inference(
        &mut self,
        peer_id: PeerId,
        session_id: [u8; 16],
        tensor_data: Vec<u8>,
        dims: Vec<usize>,
    ) -> anyhow::Result<(Vec<u8>, Vec<usize>)> {
        let request = PolygoneRequest::PetalsInfer {
            session_id,
            layer_start: self.layer_start,
            layer_end: self.layer_end,
            tensor_data,
            dims,
        };

        let response_rx = self.p2p_node.send_request(peer_id, request);
        
        match response_rx.await {
            Ok(PolygoneResponse::PetalsInfer { 
                success: true, 
                tensor_data: Some(data), 
                dims: Some(out_dims) 
            }) => Ok((data, out_dims)),
            Ok(PolygoneResponse::PetalsInfer { success: false, .. }) => {
                Err(anyhow::anyhow!("Inference failed on peer"))
            }
            Ok(_) => Err(anyhow::anyhow!("Unexpected response type")),
            Err(_) => Err(anyhow::anyhow!("Request timeout")),
        }
    }

    /// Handle incoming network events
    pub async fn handle_events<F>(
        &mut self,
        mut compute_fn: F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(&[u8], Vec<usize>, u32, u32) -> anyhow::Result<(Vec<u8>, Vec<usize>>,
    {
        while let Some(event) = self.event_rx.recv().await {
            match event {
                NetworkEvent::IncomingRequest { peer_id, request, channel } => {
                    self.handle_inference_request(peer_id, request, channel, &mut compute_fn).await?;
                }
                NetworkEvent::PeerConnected { peer_id } => {
                    info!("Peer connected: {}", peer_id);
                }
                NetworkEvent::PeerDisconnected { peer_id } => {
                    info!("Peer disconnected: {}", peer_id);
                }
                NetworkEvent::GossipReceived { topic, message, source } => {
                    debug!("Gossip on {} from {:?}: {:?}", topic, source, message);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Handle an inference request
    async fn handle_inference_request<F>(
        &mut self,
        peer_id: PeerId,
        request: PolygoneRequest,
        channel: libp2p::request_response::ResponseChannel<PolygoneResponse>,
        compute_fn: &mut F,
    ) -> anyhow::Result<()>
    where
        F: FnMut(&[u8], Vec<usize>, u32, u32) -> anyhow::Result<(Vec<u8>, Vec<usize>>,
    {
        match request {
            PolygoneRequest::PetalsInfer { session_id, layer_start, layer_end, tensor_data, dims } => {
                info!(
                    "Inference request from {}: session {:?}, layers {}-{}",
                    peer_id, session_id, layer_start, layer_end
                );

                // Track session
                self.active_sessions.insert(session_id, SessionInfo {
                    start_time: std::time::Instant::now(),
                    peer_id,
                });

                // Perform computation
                let response = match compute_fn(&tensor_data, dims, layer_start, layer_end) {
                    Ok((output_data, out_dims)) => PolygoneResponse::PetalsInfer {
                        success: true,
                        tensor_data: Some(output_data),
                        dims: Some(out_dims),
                    },
                    Err(e) => {
                        error!("Inference failed: {}", e);
                        PolygoneResponse::PetalsInfer {
                            success: false,
                            tensor_data: None,
                            dims: None,
                        }
                    }
                };

                // Clean up session
                self.active_sessions.remove(&session_id);

                // Send response
                self.p2p_node.send_response(channel, response)?;
            }
            _ => {
                warn!("Received non-Petals request from {}", peer_id);
            }
        }
        Ok(())
    }
}

/// Legacy Petals behaviour (for migration compatibility)
#[derive(NetworkBehaviour)]
pub struct PetalsBehaviour {
    pub kademlia: kad::Kademlia<kad::store::MemoryStore>,
    pub identify: identify::Behaviour,
    pub request_response: request_response::cbor::Behaviour<InferenceRequest, InferenceResponse>,
}

/// Legacy swarm builder (deprecated - use PetalsNetwork instead)
pub fn build_swarm(keypair: libp2p::identity::Keypair) -> anyhow::Result<libp2p::Swarm<PetalsBehaviour>> {
    let local_peer_id = libp2p::PeerId::from(keypair.public());
    
    let store = kad::store::MemoryStore::new(local_peer_id);
    let mut kad_config = kad::KademliaConfig::default();
    kad_config.set_protocol_names(vec![StreamProtocol::new("/pg-petals/kad/1.0.0")]);
    let mut kademlia = kad::Kademlia::with_config(local_peer_id, store, kad_config);
    kademlia.set_mode(Some(kad::Mode::Server));

    let identify = identify::Behaviour::new(identify::Config::new(
        "/pg-petals/id/1.0.0".into(),
        keypair.public(),
    ));

    let protocols = [(StreamProtocol::new("/pg-petals/infer/1.0.0"), request_response::ProtocolSupport::Full)];
    let cfg = request_response::Config::default();
    let request_response = request_response::cbor::Behaviour::new(protocols, cfg);

    let behaviour = PetalsBehaviour {
        kademlia,
        identify,
        request_response,
    };

    let swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default(),
            libp2p::noise::Config::new,
            libp2p::yamux::Config::default,
        )?
        .with_behaviour(|_| behaviour)?
        .with_swarm_config(|c| c.with_idle_connection_timeout(std::time::Duration::from_secs(60)))
        .build();

    Ok(swarm)
}
