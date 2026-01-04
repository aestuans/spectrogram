use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use js_sys::Array;
use rustfft::{num_complex::Complex, FftPlanner};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    window, AudioContext, AudioDestinationNode, AudioNode, AudioProcessingEvent, HtmlAudioElement,
    HtmlElement, HtmlMediaElement, MediaElementAudioSourceNode, MediaStream,
    MediaStreamAudioSourceNode, MediaStreamConstraints, MediaStreamTrack, ScriptProcessorNode,
};

use crate::render::{RenderBand, RenderFrame};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = window, js_name = nextTrack)]
    fn next_track();
}

pub const DEFAULT_SAMPLE_RATE: f32 = 44_100.0;

const FFT_SIZE: usize = 2048;
const HOP_SIZE: usize = 512;
const FFT_BINS: usize = FFT_SIZE / 2;
const BAND_COUNT: usize = 52;
const HISTORY_LEN: usize = 128;
const MIN_FREQ: f32 = 20.0;
const MIN_DB: f32 = -80.0;
const MAX_DB: f32 = 0.0;
const SMOOTHING: f32 = 0.8;
const ENERGY_FLOOR: f32 = 0.06; // normalized
const POWER_CURVE: f32 = 2.0;

pub struct Spectrogram {
    sample_rate: f32,
    fft: Arc<dyn rustfft::Fft<f32>>,
    fft_buffer: Vec<Complex<f32>>,
    magnitudes: Vec<f32>,
    hann_window: Vec<f32>,
    fft_normalizer: f32,

    ring_buffer: Vec<f32>,
    ring_head: usize,
    ring_len: usize,

    /// start/end FFT bins for each display band.
    band_edges: Vec<(usize, usize)>,
    /// Per-band history.
    histories: Vec<HistoryBuffer>,
    /// Last value per band.
    last_values: Vec<f32>,
}

impl Spectrogram {
    pub fn new(sample_rate: Option<f32>) -> Self {
        debug_assert!(HOP_SIZE <= FFT_SIZE);
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let hann_window = build_hann_window(FFT_SIZE);
        let window_sum: f32 = hann_window.iter().sum();
        let coherent_gain = window_sum / FFT_SIZE as f32;
        let fft_normalizer = (FFT_SIZE as f32 * 0.5) * coherent_gain;
        let band_edges = compute_band_edges(sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE));
        let fft_buffer = vec![Complex { re: 0.0, im: 0.0 }; FFT_SIZE];
        let magnitudes = vec![0.0; FFT_BINS];
        let histories = (0..BAND_COUNT)
            .map(|_| HistoryBuffer::new(HISTORY_LEN))
            .collect();
        let last_values = vec![0.0; BAND_COUNT];

        Self {
            sample_rate: sample_rate.unwrap_or(DEFAULT_SAMPLE_RATE),
            fft,
            fft_buffer,
            magnitudes,
            hann_window,
            fft_normalizer,
            ring_buffer: vec![0.0; FFT_SIZE],
            ring_head: 0,
            ring_len: 0,
            band_edges,
            histories,
            last_values,
        }
    }

    pub fn update_sample_rate(&mut self, sample_rate: f32) {
        if self.sample_rate == sample_rate {
            return;
        }
        self.sample_rate = sample_rate;
        self.band_edges = compute_band_edges(sample_rate);
    }

    pub fn reset(&mut self) {
        self.ring_buffer.fill(0.0);
        self.ring_head = 0;
        self.ring_len = 0;
        for value in &mut self.fft_buffer {
            value.re = 0.0;
            value.im = 0.0;
        }
        self.magnitudes.fill(0.0);
        for history in &mut self.histories {
            history.clear();
        }
        self.last_values.fill(0.0);
    }

    pub fn process_samples(&mut self, samples: &[f32]) {
        if !self.push_samples(samples) {
            return;
        }

        self.load_latest_frame();
        self.fft.process(&mut self.fft_buffer);
        self.compute_magnitudes();
        self.update_band_histories();
    }

    pub fn render_frame(&self) -> RenderFrame {
        let bands = self
            .histories
            .iter()
            .enumerate()
            .map(|(index, samples)| RenderBand {
                index,
                samples: samples.as_ordered_vec(),
            })
            .collect();

        RenderFrame { bands }
    }

    fn push_history(&mut self, band: usize, value: f32) {
        if let Some(history) = self.histories.get_mut(band) {
            history.push(value);
        }
    }

    fn push_samples(&mut self, samples: &[f32]) -> bool {
        if samples.is_empty() {
            return false;
        }

        for &sample in samples {
            self.ring_buffer[self.ring_head] = sample;
            self.ring_head = (self.ring_head + 1) % FFT_SIZE;
            if self.ring_len < FFT_SIZE {
                self.ring_len += 1;
            }
        }

        self.ring_len >= FFT_SIZE
    }

    fn load_latest_frame(&mut self) {
        // ring_head points at the oldest sample once the buffer is full.
        for i in 0..FFT_SIZE {
            let sample = self.ring_buffer[(self.ring_head + i) % FFT_SIZE];
            let windowed = sample * self.hann_window[i];
            self.fft_buffer[i].re = windowed;
            self.fft_buffer[i].im = 0.0;
        }
    }

    fn compute_magnitudes(&mut self) {
        for (value, magnitude) in self
            .fft_buffer
            .iter()
            .take(FFT_BINS)
            .zip(self.magnitudes.iter_mut())
        {
            *magnitude = (value.re * value.re + value.im * value.im).sqrt();
        }
    }

    fn update_band_histories(&mut self) {
        for band in 0..self.band_edges.len() {
            let (start_bin, end_bin) = self.band_edges[band];
            let mut max_value: f32 = 0.0;
            for bin in start_bin..end_bin {
                max_value = max_value.max(self.magnitudes[bin]);
            }

            let normalized = normalize_magnitude(max_value / self.fft_normalizer);
            let gated = if normalized < ENERGY_FLOOR {
                0.0
            } else {
                normalized
            };

            let smoothed = self.last_values[band] * SMOOTHING + gated * (1.0 - SMOOTHING);
            self.last_values[band] = smoothed;
            self.push_history(band, smoothed);
        }
    }
}

fn build_processor(
    context: &AudioContext,
    spectrogram: Rc<RefCell<Spectrogram>>,
) -> Result<
    (
        ScriptProcessorNode,
        Closure<dyn FnMut(AudioProcessingEvent)>,
    ),
    JsValue,
> {
    let processor = context
        .create_script_processor_with_buffer_size_and_number_of_input_channels_and_number_of_output_channels(
            HOP_SIZE as u32,
            1,
            1,
        )?;

    let silence = Rc::new(RefCell::new(vec![0.0; HOP_SIZE]));
    let silence_for_cb = silence.clone();
    let spectrogram_for_cb = spectrogram.clone();

    let callback = Closure::wrap(Box::new(move |event: AudioProcessingEvent| {
        if let Ok(input_buffer) = event.input_buffer() {
            if let Ok(samples) = input_buffer.get_channel_data(0) {
                if let Ok(mut spectrogram) = spectrogram_for_cb.try_borrow_mut() {
                    spectrogram.process_samples(&samples);
                }
            }
        }

        if let Ok(output_buffer) = event.output_buffer() {
            if output_buffer.number_of_channels() > 0 {
                // Keep the node audible-silent while still connected to the destination.
                let length = output_buffer.length() as usize;
                let mut silence = silence_for_cb.borrow_mut();
                if silence.len() != length {
                    silence.resize(length, 0.0);
                } else {
                    silence.fill(0.0);
                }
                let _ = output_buffer.copy_to_channel(&silence, 0);
            }
        }
    }) as Box<dyn FnMut(AudioProcessingEvent)>);

    processor.set_onaudioprocess(Some(callback.as_ref().unchecked_ref()));

    Ok((processor, callback))
}

pub struct MicrophonePipeline {
    _context: AudioContext,
    _stream: MediaStream,
    _source: MediaStreamAudioSourceNode,
    _processor: ScriptProcessorNode,
    _callback: Closure<dyn FnMut(AudioProcessingEvent)>,
}

impl MicrophonePipeline {
    pub async fn new_with_context(
        context: AudioContext,
        spectrogram: Rc<RefCell<Spectrogram>>,
    ) -> Result<Self, JsValue> {
        let window = window().ok_or_else(|| JsValue::from_str("No window available"))?;
        let navigator = window.navigator();
        let media_devices = navigator.media_devices()?;
        let constraints = MediaStreamConstraints::new();
        constraints.set_audio(&JsValue::TRUE);

        let stream_promise = media_devices.get_user_media_with_constraints(&constraints)?;
        let stream_value = JsFuture::from(stream_promise).await?;
        let stream: MediaStream = stream_value.dyn_into()?;

        {
            let mut spectrogram = spectrogram.borrow_mut();
            spectrogram.update_sample_rate(context.sample_rate());
        }

        let source = context.create_media_stream_source(&stream)?;
        let (processor, callback) = build_processor(&context, spectrogram.clone())?;

        let source_node: &AudioNode = source.unchecked_ref();
        let processor_node: &AudioNode = processor.unchecked_ref();
        source_node.connect_with_audio_node(processor_node)?;

        let destination: AudioDestinationNode = context.destination();
        let destination_node: &AudioNode = destination.unchecked_ref();
        processor_node.connect_with_audio_node(destination_node)?;

        Ok(Self {
            _context: context,
            _stream: stream,
            _source: source,
            _processor: processor,
            _callback: callback,
        })
    }

    pub fn shutdown(&self) {
        self._processor.set_onaudioprocess(None);
        let _ = self._processor.disconnect();
        let _ = self._source.disconnect();
        let tracks: Array = self._stream.get_tracks();
        for index in 0..tracks.length() {
            let track_value = tracks.get(index);
            if let Ok(track) = track_value.dyn_into::<MediaStreamTrack>() {
                track.stop();
            }
        }
        let _ = self._context.close();
    }
}

pub struct MusicPipeline {
    _context: AudioContext,
    _audio: HtmlAudioElement,
    _source: MediaElementAudioSourceNode,
    _processor: ScriptProcessorNode,
    _callback: Closure<dyn FnMut(AudioProcessingEvent)>,
    _ended_callback: Closure<dyn FnMut()>,
}

impl MusicPipeline {
    pub async fn new_with_context_and_url(
        context: AudioContext,
        spectrogram: Rc<RefCell<Spectrogram>>,
        url: &str,
    ) -> Result<Self, JsValue> {
        let audio = HtmlAudioElement::new()?;
        let media_element: &HtmlMediaElement = audio.unchecked_ref();
        media_element.set_preload("auto");
        media_element.set_loop(false);
        media_element.set_src(url);
        let html_element: &HtmlElement = audio.unchecked_ref();
        let ended_callback = Closure::wrap(Box::new(move || {
            next_track();
        }) as Box<dyn FnMut()>);
        html_element.set_onended(Some(ended_callback.as_ref().unchecked_ref()));

        {
            let mut spectrogram = spectrogram.borrow_mut();
            spectrogram.update_sample_rate(context.sample_rate());
        }

        let source = context.create_media_element_source(media_element)?;

        let (processor, callback) = build_processor(&context, spectrogram.clone())?;

        let source_node: &AudioNode = source.unchecked_ref();
        let processor_node: &AudioNode = processor.unchecked_ref();
        source_node.connect_with_audio_node(processor_node)?;

        let destination: AudioDestinationNode = context.destination();
        let destination_node: &AudioNode = destination.unchecked_ref();
        source_node.connect_with_audio_node(destination_node)?;
        processor_node.connect_with_audio_node(destination_node)?;

        let play_promise = media_element.play()?;
        let _ = JsFuture::from(play_promise).await?;

        Ok(Self {
            _context: context,
            _audio: audio,
            _source: source,
            _processor: processor,
            _callback: callback,
            _ended_callback: ended_callback,
        })
    }

    pub fn shutdown(&self) {
        let html_element: &HtmlElement = self._audio.unchecked_ref();
        html_element.set_onended(None);
        self._processor.set_onaudioprocess(None);
        let _ = self._processor.disconnect();
        let _ = self._source.disconnect();
        let media_element: &HtmlMediaElement = self._audio.unchecked_ref();
        let _ = media_element.pause();
        media_element.set_src("");
        media_element.load();
        let _ = self._context.close();
    }
}

fn build_hann_window(size: usize) -> Vec<f32> {
    if size <= 1 {
        return vec![1.0];
    }
    let denom = (size - 1) as f32;
    let mut window = Vec::with_capacity(size);
    for i in 0..size {
        let phase = 2.0 * std::f32::consts::PI * i as f32 / denom;
        window.push(0.5 - 0.5 * phase.cos());
    }
    window
}

fn compute_band_edges(sample_rate: f32) -> Vec<(usize, usize)> {
    let nyquist = (sample_rate * 0.5).max(MIN_FREQ + 1.0);
    let min_freq = MIN_FREQ.min(nyquist * 0.9);
    let log_min = min_freq.ln();
    let log_max = nyquist.ln();
    let bin_scale = FFT_SIZE as f32 / sample_rate;

    let mut edges = Vec::with_capacity(BAND_COUNT);
    for band in 0..BAND_COUNT {
        let t0 = band as f32 / BAND_COUNT as f32;
        let t1 = (band + 1) as f32 / BAND_COUNT as f32;
        let f0 = (log_min + (log_max - log_min) * t0).exp();
        let f1 = (log_min + (log_max - log_min) * t1).exp();

        let mut start = (f0 * bin_scale).floor() as usize;
        let mut end = (f1 * bin_scale).ceil() as usize;
        start = start.min(FFT_BINS.saturating_sub(1));
        if end <= start {
            end = start + 1;
        }
        end = end.min(FFT_BINS);

        edges.push((start, end));
    }

    edges
}

fn normalize_magnitude(value: f32) -> f32 {
    let db = 20.0 * value.max(1e-12).log10();
    let normalized = ((db - MIN_DB) / (MAX_DB - MIN_DB)).clamp(0.0, 1.0);
    normalized.powf(POWER_CURVE)
}

struct HistoryBuffer {
    data: Vec<f32>,
    head: usize,
    filled: usize,
}

impl HistoryBuffer {
    fn new(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
            head: 0,
            filled: 0,
        }
    }

    fn push(&mut self, value: f32) {
        if self.data.is_empty() {
            return;
        }
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.data.len();
        if self.filled < self.data.len() {
            self.filled += 1;
        }
    }

    fn clear(&mut self) {
        self.data.fill(0.0);
        self.head = 0;
        self.filled = 0;
    }

    fn as_ordered_vec(&self) -> Vec<f32> {
        let len = self.data.len();
        if len == 0 {
            return Vec::new();
        }

        let mut ordered = vec![0.0; len];
        if self.filled == 0 {
            return ordered;
        }

        if self.filled < len {
            let offset = len - self.filled;
            ordered[offset..].copy_from_slice(&self.data[..self.filled]);
            return ordered;
        }

        let tail = len - self.head;
        ordered[..tail].copy_from_slice(&self.data[self.head..]);
        ordered[tail..].copy_from_slice(&self.data[..self.head]);
        ordered
    }
}
