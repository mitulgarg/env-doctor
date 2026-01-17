# Recording Demos

This guide explains how to record terminal demos for the documentation.

## Asciinema (Recommended)

[Asciinema](https://asciinema.org) creates lightweight, copy-pasteable terminal recordings.

### Installation

```bash
# macOS
brew install asciinema

# Ubuntu/Debian
sudo apt install asciinema

# pip
pip install asciinema
```

### Recording

```bash
# Start recording
asciinema rec demo.cast

# Run your commands...
env-doctor check
env-doctor model llama-3-8b

# Press Ctrl+D or type 'exit' to stop
```

### Tips for Good Recordings

1. **Clear screen first**: Start with `clear`
2. **Type slowly**: Pause between commands for readability
3. **Keep it short**: 15-30 seconds is ideal
4. **Show one feature**: Focus on a single command per recording

### Uploading

```bash
# Upload to asciinema.org
asciinema upload demo.cast

# Or authenticate first for your account
asciinema auth
asciinema upload demo.cast
```

### Embedding in Docs

After uploading, you'll get a URL like `https://asciinema.org/a/123456`.

**In Markdown:**

```markdown
[![asciicast](https://asciinema.org/a/123456.svg)](https://asciinema.org/a/123456)
```

**As JavaScript embed (auto-plays):**

```html
<script src="https://asciinema.org/a/123456.js" id="asciicast-123456" async data-autoplay="true" data-speed="1.5"></script>
```

## Recording Checklist

Before recording:

- [ ] Clean terminal (no sensitive info)
- [ ] Clear history if needed
- [ ] Set consistent terminal size (80x24 recommended)
- [ ] Prepare commands to run

Recording:

- [ ] Clear screen
- [ ] Type commands clearly
- [ ] Pause to let output be readable
- [ ] Keep under 30 seconds

After recording:

- [ ] Review the recording
- [ ] Re-record if needed
- [ ] Upload and get embed code