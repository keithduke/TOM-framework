# Security Best Practices

## Overview

TOM is designed to be secure by default while remaining flexible for power users.
All processing happens locally on your machine - no data leaves your device.

## File Reading Security

### Allowed Paths

By default, the `read` tool can only access files in:
- Your home directory (`~`)
- Current working directory
- `/tmp` (temporary files)

### Blocked Files

The following file types are automatically blocked:
- Environment files (`.env`, `.env.local`, etc.)
- Credentials (`credentials.json`, `secrets.yaml`)
- SSH keys (`id_rsa`, `id_ed25519`, etc.)
- Private keys (`.pem`, `.key` files)
- AWS/Azure/GCloud credentials

### Path Traversal Protection

TOM validates all file paths and prevents access outside allowed directories.
Even if you try to read `/etc/passwd`, the request will be denied.

## Network Security

### Local-Only by Default

- TOM's API server binds to `127.0.0.1` (localhost only)
- No external network access by default
- All communication over loopback interface

### Exposing the API (Advanced)

If you want to access TOM from other devices:

```bash
# ⚠️ WARNING: Only do this on trusted networks
python main.py --host 0.0.0.0 --port 8000
```

**Security recommendations:**
- Use a firewall to limit access
- Set up API key authentication (future feature)
- Use HTTPS/TLS in production
- Never expose to the internet without authentication

## Data Privacy

### What's Stored

- **Conversation history**: In-memory only (lost on restart)
- **Cache files**: Model KV cache (no user data)
- **Logs**: System events, no message content

### What's NOT Stored

- User messages are never written to disk
- No telemetry or analytics
- No external API calls

## Best Practices

1. **Run with least privilege**: Don't run TOM as root/admin
2. **Review logs**: Check `~/.tom/` for any suspicious activity
3. **Keep updated**: Update dependencies regularly
4. **Trust but verify**: Review tool calls before approving (when that feature is added)

## Reporting Security Issues

If you discover a security vulnerability, please email:
[your-email@example.com]

Do not open public issues for security problems.
