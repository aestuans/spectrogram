use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{AudioContext, BaseAudioContext};

mod audio;
mod render;

use crate::audio::{MicrophonePipeline, MusicPipeline, Spectrogram};
use crate::render::Renderer;

const MUSIC_SAMPLE_URL: &str = "assets/Winter-Vivaldi-PM-Music.mp3";

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum SourceKind {
    Microphone,
    Music,
}

enum ActiveAudio {
    Microphone(MicrophonePipeline),
    Music(MusicPipeline),
}

impl ActiveAudio {
    fn shutdown(&self) {
        match self {
            ActiveAudio::Microphone(pipeline) => pipeline.shutdown(),
            ActiveAudio::Music(pipeline) => pipeline.shutdown(),
        }
    }
}

struct AppState {
    renderer: Renderer,
    spectrogram: Rc<RefCell<Spectrogram>>,
    audio: Option<ActiveAudio>,
    active_source: Option<SourceKind>,
    active_music_url: Option<String>,
}

thread_local! {
    static APP_STATE: RefCell<Option<Rc<RefCell<AppState>>>> = RefCell::new(None);
}

/// Application entry point called by wasm-bindgen.
#[wasm_bindgen(start)]
pub fn run_app() -> Result<(), JsValue> {
    let renderer = Renderer::new().map_err(|e| JsValue::from_str(&e))?;
    let spectrogram = Rc::new(RefCell::new(Spectrogram::new(None)));
    let app_state = Rc::new(RefCell::new(AppState {
        renderer,
        spectrogram: spectrogram.clone(),
        audio: None,
        active_source: None,
        active_music_url: None,
    }));

    APP_STATE.with(|state| {
        *state.borrow_mut() = Some(app_state.clone());
    });

    start_render_loop(app_state.clone())?;

    Ok(())
}

#[wasm_bindgen]
pub async fn start_microphone() -> Result<(), JsValue> {
    start_source(SourceKind::Microphone, None).await
}

#[wasm_bindgen]
pub async fn start_music() -> Result<(), JsValue> {
    start_source(SourceKind::Music, None).await
}

#[wasm_bindgen]
pub async fn start_music_with_url(url: String) -> Result<(), JsValue> {
    start_source(SourceKind::Music, Some(url)).await
}

#[wasm_bindgen]
pub fn resize_canvas() -> Result<(), JsValue> {
    let app_state = get_app_state()?;
    {
        let app = app_state.borrow();
        app.renderer
            .resize()
            .map_err(|e| JsValue::from_str(&e))?;
    }
    Ok(())
}

async fn start_source(kind: SourceKind, music_url: Option<String>) -> Result<(), JsValue> {
    let app_state = get_app_state()?;

    let resolved_music_url = match kind {
        SourceKind::Music => Some(music_url.unwrap_or_else(|| MUSIC_SAMPLE_URL.to_string())),
        SourceKind::Microphone => None,
    };

    let already_active = {
        let state = app_state.borrow();
        match kind {
            SourceKind::Microphone => state.active_source == Some(SourceKind::Microphone),
            SourceKind::Music => {
                if state.active_source != Some(SourceKind::Music) {
                    false
                } else {
                    state.active_music_url.as_deref() == resolved_music_url.as_deref()
                }
            }
        }
    };
    if already_active {
        return Ok(());
    }

    let spectrogram = {
        let mut state = app_state.borrow_mut();
        if let Some(audio) = state.audio.take() {
            audio.shutdown();
        }
        state.active_source = None;
        state.active_music_url = None;
        state.spectrogram.borrow_mut().reset();
        state.spectrogram.clone()
    };

    let context = AudioContext::new()?;
    // Resuming the context after a user gesture avoids autoplay restrictions.
    if let Some(base) = context.dyn_ref::<BaseAudioContext>() {
        let _ = base.resume();
    }

    let pipeline = match kind {
        SourceKind::Microphone => MicrophonePipeline::new_with_context(context, spectrogram)
            .await
            .map(ActiveAudio::Microphone)?,
        SourceKind::Music => {
            let url = resolved_music_url.as_deref().unwrap_or(MUSIC_SAMPLE_URL);
            MusicPipeline::new_with_context_and_url(context, spectrogram, url)
                .await
                .map(ActiveAudio::Music)?
        }
    };

    {
        let mut state = app_state.borrow_mut();
        state.audio = Some(pipeline);
        state.active_source = Some(kind);
        state.active_music_url = resolved_music_url;
    }

    Ok(())
}

fn get_app_state() -> Result<Rc<RefCell<AppState>>, JsValue> {
    APP_STATE.with(|state| {
        state
            .borrow()
            .clone()
            .ok_or_else(|| JsValue::from_str("App state not initialized."))
    })
}

fn start_render_loop(app_state: Rc<RefCell<AppState>>) -> Result<(), JsValue> {
    let frame_handle: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> =
        Rc::new(RefCell::new(None));
    let frame_handle_clone = frame_handle.clone();

    *frame_handle_clone.borrow_mut() = Some(Closure::wrap(Box::new(move |_time| {
        {
            let app = app_state.borrow();
            let frame = app.spectrogram.borrow().render_frame();
            let _ = app.renderer.render(&frame);
        }

        if let Some(callback) = frame_handle.borrow().as_ref() {
            let _ = request_animation_frame(callback);
        }
    }) as Box<dyn FnMut(f64)>));

    if let Some(callback) = frame_handle_clone.borrow().as_ref() {
        request_animation_frame(callback)?;
    }

    Ok(())
}

fn request_animation_frame(callback: &Closure<dyn FnMut(f64)>) -> Result<i32, JsValue> {
    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window available"))?;
    window.request_animation_frame(callback.as_ref().unchecked_ref())
}
