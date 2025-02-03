# Captain

Watches your screen and uses LLMs to help with your work.

## Requirements

- [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) and [Rust](https://www.rust-lang.org/tools/install)

## Chat

Run a chat conversation with the LLM. It spins off a thread that will watch your screen.

```bash
cargo run -- shell
```

- The last ~3 minutes of your screen activity will be explicitly captured.
- From ~3 minutes to the start of the program run, the model will search for the relevant events in the trajectory via dense embeddings.

To reduce the amount of data that is stored in the trajectory, an async task worker removes redundant, similar, or subset images that likely do not contribute to the trajectory.
In the future, async workers will perform image merging (for overlayed images), image trimming (e.g. computer frames), and other creative ways to reduce the amount of data stored in the trajectory.

## Autocomplete

**Warning:** This tool is an experimental feature. It will directly send keyboard events to your computer.

```bash
cargo run -- autocomplete
```

Tap `Cmd` to trigger an autocompletion. Tap `Ctrl` to cancel.
