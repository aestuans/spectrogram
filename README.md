# spectrogram

A spectrogram with the aesthetic styling of the Unknown Pleasures album art (originally the signals from pulsar CP1919).

Written in Rust and rendered with WebGL.

https://aestuans.github.io/spectrogram

## Build

1. Install `wasm-pack`
```
cargo install wasm-pack
```

2. Generate the wasm module

```
wasm-pack build --release --target web --out-dir dist/
```

4. Run with a simple http server. Such as:
```
python -m http.server
```

## Credits

[Music Credits](https://aestuans.github.io/spectrogram/music-credits.html)