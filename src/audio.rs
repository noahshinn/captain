use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION};
use reqwest::multipart::{Form, Part};
use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TranscriptionFromFilePathError {
    #[error("invalid file path: {0}")]
    InvalidFilePath(#[from] std::io::Error),
    #[error("unsupported file type: {0}")]
    UnsupportedFileType(String),
    #[error("transcription error")]
    TranscriptionError(#[from] TranscriptionError),
}

#[derive(Error, Debug)]
pub enum TranscriptionError {
    #[error("authorization error: OPENAI_API_KEY is not set")]
    AuthorizationError,
    #[error("api error")]
    ApiError(#[from] reqwest::Error),
    #[error("invalid file path")]
    InvalidFilePath(#[from] std::io::Error),
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
}

const OPENAI_TRANSCRIPTION_URL: &str = "https://api.openai.com/v1/audio/transcriptions";

pub async fn transcribe_audio(audio_data: Vec<u8>) -> Result<String, TranscriptionError> {
    let api_key = match env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => return Err(TranscriptionError::AuthorizationError),
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        AUTHORIZATION,
        match HeaderValue::from_str(&format!("Bearer {api_key}")) {
            Ok(value) => value,
            Err(_) => return Err(TranscriptionError::AuthorizationError),
        },
    );

    let file_part = Part::bytes(audio_data)
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| TranscriptionError::ApiError(e.into()))?;

    let form = Form::new()
        .part("file", file_part)
        .text("model", "whisper-1")
        .text("response_format", "json");

    let client = reqwest::Client::new();
    let response = match client
        .post(OPENAI_TRANSCRIPTION_URL)
        .headers(headers)
        .multipart(form)
        .send()
        .await
    {
        Ok(response) => response,
        Err(e) => return Err(TranscriptionError::ApiError(e)),
    };

    let response_body: TranscriptionResponse = match response.json().await {
        Ok(body) => body,
        Err(e) => return Err(TranscriptionError::ApiError(e)),
    };

    Ok(response_body.text)
}

pub async fn transcribe_audio_from_file_path<P: AsRef<Path>>(
    file_path: P,
) -> Result<String, TranscriptionFromFilePathError> {
    let mut file = match File::open(file_path) {
        Ok(file) => file,
        Err(e) => return Err(TranscriptionFromFilePathError::InvalidFilePath(e)),
    };
    let mut buffer = Vec::new();
    if let Err(e) = file.read_to_end(&mut buffer) {
        return Err(TranscriptionFromFilePathError::InvalidFilePath(e));
    }
    match transcribe_audio(buffer).await {
        Ok(text) => Ok(text),
        Err(e) => Err(TranscriptionFromFilePathError::TranscriptionError(e)),
    }
}
