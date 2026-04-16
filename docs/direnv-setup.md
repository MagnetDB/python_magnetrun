# direnv Setup

[direnv](https://direnv.net/) is used to automatically export project secrets
(API keys, server addresses, …) into the shell whenever you enter the project
directory, without storing them in version control.

## Prerequisites

Install direnv and hook it into your shell:

```bash
# Debian / Ubuntu
sudo apt install direnv

```

Then add the hook to your shell rc file (run once):

| Shell | Command |
|-------|---------|
| bash | `echo 'eval "$(direnv hook bash)"' >> ~/.bashrc` |

Reload your shell afterwards (`exec $SHELL`).

## First-time project setup

```bash
# 1. Copy the example env file
cp .env.example .env

# 2. Fill in your secrets
$EDITOR .env

# 3. Allow direnv to load the file (required after any change to .envrc)
direnv allow
```

`.env` is listed in `.gitignore` and will never be committed.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `USERDB_SERVER` | UserDB API server hostname / IP | `147.173.81.141` |
| `USERDB_API_KEY` | Bearer token for the UserDB API | *(required)* |
| `MAGNETDB_SERVER` | MagnetDB API server hostname / IP | *(required)* |
| `MAGNETDB_API_KEY` | Bearer token for the MagnetDB API | *(required)* |
| `RUSTFS_ENDPOINT` | RustFS / S3-compatible endpoint URL | `http://localhost:9000` |
| `ACCESS_KEY` | S3 access key | *(required)* |
| `SECRET_KEY` | S3 secret key | *(required)* |

## Day-to-day use

Once set up, direnv loads `.env` automatically on every `cd` into the project:

```
direnv: loading ~/github/python_magnetrun/.env
direnv: export +USERDB_SERVER +USERDB_API_KEY +MAGNETDB_SERVER +MAGNETDB_API_KEY ...
```

To temporarily disable: `direnv deny` / re-enable: `direnv allow`.

## Updating secrets

Edit `.env`, then run `direnv reload` (or simply `cd` out and back in).
