use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::{
    Json, Router,
    extract::State,
    response::Html,
    routing::{get, post},
};
use rmcp::{
    RoleClient, ServiceExt,
    model::Tool,
    transport::{
        IntoTransport, StreamableHttpClientTransport,
        streamable_http_client::StreamableHttpClientTransportConfig,
    },
};
use serde::{Deserialize, Serialize};

const DEFAULT_SERVER_URL: &str = "https://mcp.deepwiki.com/mcp";
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

struct AppState {
    default_url: String,
}

#[derive(Serialize, Debug)]
struct ServerSummary {
    name: Option<String>,
    title: Option<String>,
    version: Option<String>,
    protocol_version: String,
    instructions: Option<String>,
    capabilities: serde_json::Value,
}

#[derive(Serialize, Debug)]
struct ToolSummary {
    name: String,
    title: Option<String>,
    description: Option<String>,
    input_schema: serde_json::Value,
}

impl From<Tool> for ToolSummary {
    fn from(tool: Tool) -> Self {
        ToolSummary {
            name: tool.name.to_string(),
            title: tool.title,
            description: tool.description.map(|d| d.to_string()),
            input_schema: serde_json::Value::Object((*tool.input_schema).clone()),
        }
    }
}

/// What one connect-and-list round trip produced.
#[derive(Serialize, Debug)]
struct Inspection {
    server: ServerSummary,
    tools: Vec<ToolSummary>,
    handshake_ms: u128,
    list_tools_ms: u128,
}

/// The whole lesson in one function: open an MCP session over whatever
/// transport it's given (`initialize` -> `notifications/initialized`), read
/// what the server said about itself, page through `tools/list`, and close
/// the session cleanly. Generic over the transport so the tests can drive it
/// against an in-memory server instead of the network.
async fn inspect<T, E, A>(transport: T) -> Result<Inspection, String>
where
    T: IntoTransport<RoleClient, E, A>,
    E: std::error::Error + Send + Sync + 'static,
{
    let started = Instant::now();
    // `()` is the SDK's no-op client handler: we only send requests, we
    // don't need to answer any server-initiated ones (sampling, roots, ...).
    let client = ()
        .serve(transport)
        .await
        .map_err(|e| format!("MCP handshake failed: {e}"))?;
    let handshake_ms = started.elapsed().as_millis();

    let info = client
        .peer_info()
        .ok_or_else(|| "Connected, but the server sent no initialize result".to_string())?;
    let server = ServerSummary {
        name: info.server_info.as_ref().map(|s| s.name.clone()),
        title: info.server_info.as_ref().and_then(|s| s.title.clone()),
        version: info.server_info.as_ref().map(|s| s.version.clone()),
        protocol_version: info.protocol_version.to_string(),
        instructions: info.instructions.clone(),
        capabilities: serde_json::to_value(&info.capabilities).unwrap_or_default(),
    };

    let started = Instant::now();
    let tools = client.list_all_tools().await;
    let list_tools_ms = started.elapsed().as_millis();

    // Close the session regardless of how tools/list went; a failure here
    // doesn't change what we already learned.
    let _ = client.cancel().await;

    let tools = tools.map_err(|e| format!("tools/list failed: {e}"))?;
    Ok(Inspection {
        server,
        tools: tools.into_iter().map(ToolSummary::from).collect(),
        handshake_ms,
        list_tools_ms,
    })
}

async fn inspect_http(url: &str, token: Option<&str>) -> Result<Inspection, String> {
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err("Server URL must start with http:// or https://".to_string());
    }
    let mut config = StreamableHttpClientTransportConfig::with_uri(url.to_string());
    if let Some(token) = token {
        config = config.auth_header(token.to_string());
    }
    let transport = StreamableHttpClientTransport::from_config(config);
    tokio::time::timeout(CONNECT_TIMEOUT, inspect(transport))
        .await
        .unwrap_or_else(|_| Err(format!("Timed out after {}s", CONNECT_TIMEOUT.as_secs())))
}

#[derive(Deserialize)]
struct ConnectRequest {
    url: Option<String>,
    token: Option<String>,
}

#[derive(Serialize)]
struct ConnectResponse {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Inspection>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn connect(State(state): State<Arc<AppState>>, Json(req): Json<ConnectRequest>) -> Json<ConnectResponse> {
    let url = req
        .url
        .map(|u| u.trim().to_string())
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| state.default_url.clone());
    let token = req.token.as_deref().map(str::trim).filter(|t| !t.is_empty());

    let (result, error) = match inspect_http(&url, token).await {
        Ok(inspection) => {
            println!(
                "Connected to {url}: {} tool(s), handshake {} ms",
                inspection.tools.len(),
                inspection.handshake_ms
            );
            (Some(inspection), None)
        }
        Err(e) => {
            eprintln!("Connect to {url} failed: {e}");
            (None, Some(e))
        }
    };
    Json(ConnectResponse { url, result, error })
}

#[derive(Serialize)]
struct ConfigResponse {
    default_url: String,
}

async fn config(State(state): State<Arc<AppState>>) -> Json<ConfigResponse> {
    Json(ConfigResponse {
        default_url: state.default_url.clone(),
    })
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}

#[tokio::main]
async fn main() {
    let default_url = std::env::var("MCP_SERVER_URL")
        .ok()
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| DEFAULT_SERVER_URL.to_string());
    println!("Default MCP server: {default_url}");

    let app = Router::new()
        .route("/", get(index))
        .route("/api/config", get(config))
        .route("/api/connect", post(connect))
        .with_state(Arc::new(AppState { default_url }));

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await.unwrap();
    println!("Listening on http://localhost:{port}");
    axum::serve(listener, app).await.unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmcp::{
        ErrorData, RoleServer, ServerHandler,
        model::{Implementation, ListToolsResult, PaginatedRequestParams, ServerCapabilities, ServerConfig},
        service::RequestContext,
    };

    /// A minimal MCP server with two fixed tools, served in-process.
    #[derive(Clone)]
    struct FixtureServer;

    fn schema(value: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
        value.as_object().cloned().unwrap()
    }

    impl ServerHandler for FixtureServer {
        fn get_info(&self) -> ServerConfig {
            ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
                .with_server_info(Implementation::new("fixture-server", "1.2.3"))
                .with_instructions("Test fixture")
        }

        async fn list_tools(
            &self,
            _request: Option<PaginatedRequestParams>,
            _context: RequestContext<RoleServer>,
        ) -> Result<ListToolsResult, ErrorData> {
            Ok(ListToolsResult::with_all_items(vec![
                Tool::new(
                    "echo",
                    "Echo the given text back",
                    schema(serde_json::json!({
                        "type": "object",
                        "properties": { "text": { "type": "string" } },
                        "required": ["text"]
                    })),
                ),
                Tool::new(
                    "add",
                    "Add two numbers",
                    schema(serde_json::json!({
                        "type": "object",
                        "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
                        "required": ["a", "b"]
                    })),
                ),
            ]))
        }
    }

    #[tokio::test]
    async fn connects_and_lists_tools() {
        let (server_io, client_io) = tokio::io::duplex(64 * 1024);
        let server = tokio::spawn(async move {
            let running = FixtureServer.serve(server_io).await.expect("server handshake");
            let _ = running.waiting().await;
        });

        let inspection = inspect(client_io).await.expect("inspect should succeed");

        assert_eq!(inspection.server.name.as_deref(), Some("fixture-server"));
        assert_eq!(inspection.server.version.as_deref(), Some("1.2.3"));
        assert_eq!(inspection.server.instructions.as_deref(), Some("Test fixture"));
        assert!(!inspection.server.protocol_version.is_empty());
        assert!(inspection.server.capabilities.get("tools").is_some());

        let names: Vec<&str> = inspection.tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["echo", "add"]);
        assert_eq!(inspection.tools[1].description.as_deref(), Some("Add two numbers"));
        assert_eq!(inspection.tools[0].input_schema["required"][0], "text");

        server.await.unwrap();
    }

    #[tokio::test]
    async fn handshake_failure_is_reported_as_error() {
        // The server end is dropped immediately, so initialize never gets a reply.
        let (server_io, client_io) = tokio::io::duplex(1024);
        drop(server_io);
        let err = inspect(client_io).await.unwrap_err();
        assert!(err.starts_with("MCP handshake failed"), "{err}");
    }

    #[tokio::test]
    async fn rejects_non_http_url() {
        let err = inspect_http("ftp://example.com", None).await.unwrap_err();
        assert!(err.contains("http"), "{err}");
    }
}
