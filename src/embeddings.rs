use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use std::env;

const OPENAI_API_URL: &str = "https://api.openai.com/v1/embeddings";

#[derive(Serialize)]
struct RequestBody {
    model: &'static str,
    input: Vec<String>,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

pub async fn embedding(texts: Vec<String>) -> Result<Vec<Vec<f32>>, reqwest::Error> {
    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!("Bearer {api_key}")).unwrap(),
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let req_body = RequestBody {
        model: "text-embedding-3-small",
        input: texts,
    };

    let client = reqwest::Client::new();
    let response = match client
        .post(OPENAI_API_URL)
        .headers(headers)
        .json(&req_body)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => {
            println!("Error: {}", e);
            return Err(e);
        }
    };
    let response_body: Response = match response.json().await {
        Ok(response_body) => response_body,
        Err(e) => {
            println!("Error: {}", e);
            return Err(e);
        }
    };
    Ok(response_body
        .data
        .into_iter()
        .map(|d| d.embedding)
        .collect())
}
