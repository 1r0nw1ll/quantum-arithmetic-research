# Digital Garden & Home Lab Setup Guide

This guide walks you through completing **Phase 1: Digital Garden Setup** and preparing for the multi-agent home lab infrastructure.

---

## Phase 1 Status

### ✅ Completed (Automated)

1. **Private content secured**
   - Created `private/` directory
   - Moved `patents/`, `funding/`, `papers/` into `private/`
   - Updated `.gitignore` to exclude sensitive files

2. **GitHub Pages structure created**
   - `docs/` directory with Jekyll configuration
   - `docs/index.md` - Homepage
   - `docs/quickstart.md` - Getting started guide
   - `docs/assets/images/` - Visualizations copied

3. **Documentation updated**
   - New `README.md` with digital garden info
   - Hybrid open-source model explained

### ⬜ Manual Steps Required

The following require GitHub web interface (cannot be automated):

---

## Step 1: Enable GitHub Wiki

1. Go to your repository on GitHub: `https://github.com/<your-username>/signal_experiments`

2. Click **Settings** (top right)

3. Scroll down to **Features** section

4. Check the box for **✅ Wikis**

5. Click **Save**

---

## Step 2: Enable GitHub Pages

1. In repository **Settings**, scroll to **Pages** section (left sidebar)

2. Configure:
   - **Source**: Deploy from a branch
   - **Branch**: `main`
   - **Folder**: `/docs`

3. Click **Save**

4. Wait 2-3 minutes for deployment

5. Your site will be available at: `https://<your-username>.github.io/signal_experiments/`

---

## Step 3: Create Wiki Pages

### Homepage (Home.wiki)

1. Go to Wiki tab in your repository

2. Create the first page (will be named "Home" automatically)

3. Copy content from `docs/index.md` (adjust markdown as needed)

4. Click **Save Page**

### Getting Started Page

1. Click **New Page** in Wiki

2. Title: `Getting-Started`

3. Copy content from `docs/quickstart.md`

4. Click **Save Page**

### Additional Pages to Create

Create these pages by clicking **New Page**:

**Experiments**
- Title: `Experiments`
- Content: List of all experiments with links to Python files

**Research-Findings**
- Title: `Research-Findings`
- Content: Summary of session closeouts (redact sensitive info)

**API-Documentation**
- Title: `API-Documentation`
- Content: Docstrings from `qa_core.py`, class definitions

---

## Step 4: First Commit & Push

Now that sensitive content is protected in `private/`:

```bash
# Stage changes
git add .
git add docs/
git add README.md
git add .gitignore

# Commit
git commit -m "Phase 1: Digital garden setup

- Secured private content (patents, funding, papers) in private/ directory
- Created GitHub Pages site in docs/
- Added comprehensive README with hybrid open-source model
- Prepared wiki content from GEMINI.md and QUICKSTART.md

Ready for GitHub Wiki and Pages activation."

# Push to GitHub
git push origin main
```

**⚠️ IMPORTANT**: Verify that `private/` is **NOT** being tracked:

```bash
git status
# Should NOT show private/ directory

git ls-files | grep private
# Should return nothing
```

If you see `private/` files, **STOP** and run:
```bash
git rm -r --cached private/
git commit -m "Remove private/ from tracking"
```

---

## Step 5: Verify Digital Garden

### Check GitHub Pages

1. Visit: `https://<your-username>.github.io/signal_experiments/`

2. Verify:
   - ✅ Homepage loads with QA Research title
   - ✅ Navigation links work
   - ✅ Quickstart page accessible
   - ✅ Images display (quartz_*.png)

### Check Wiki

1. Visit: `https://github.com/<your-username>/signal_experiments/wiki`

2. Verify:
   - ✅ Home page shows
   - ✅ Getting Started page exists
   - ✅ Sidebar navigation works

---

## Step 6: Share Your Digital Garden

Your research is now public! Share links:

🌐 **Website**: `https://<your-username>.github.io/signal_experiments/`
📖 **Wiki**: `https://github.com/<your-username>/signal_experiments/wiki`
💻 **Repository**: `https://github.com/<your-username>/signal_experiments`

### Update README Badges

Add these to `README.md` (replace `<your-username>`):

```markdown
[![Website](https://img.shields.io/badge/website-live-brightgreen)](https://<your-username>.github.io/signal_experiments/)
[![Wiki](https://img.shields.io/badge/wiki-documentation-blue)](https://github.com/<your-username>/signal_experiments/wiki)
```

---

## Next: Phase 2 - Docker Foundation

Once Phase 1 is complete, proceed to Docker containerization:

### Week 1: Docker Basics

1. **Install Docker Desktop**: [https://docs.docker.com/get-docker/](https://docs.docker.com/get-docker/)

2. **Complete Docker tutorial**:
   ```bash
   docker run -d -p 80:80 docker/getting-started
   # Open browser: http://localhost
   ```

3. **Verify installation**:
   ```bash
   docker --version
   docker compose version
   ```

### Week 2: First Containerized Experiment

Create `Dockerfile` for signal processing:

```dockerfile
# Dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
RUN pip install numpy matplotlib pandas scikit-learn

# Copy experiment
COPY run_signal_experiments_final.py .

# Run
CMD ["python", "run_signal_experiments_final.py"]
```

Build and run:
```bash
docker build -t qa-signal-experiment .
docker run -v $(pwd)/results:/app/results qa-signal-experiment
```

**Expected**: `results/signal_classification_results.png` created.

---

## Troubleshooting

### GitHub Pages not deploying

**Check build status**:
1. Go to **Actions** tab in repository
2. Look for "pages build and deployment" workflow
3. If failed, click for error details

**Common issues**:
- Jekyll syntax error in `_config.yml` → Check YAML formatting
- Missing front matter in markdown files → Add `---` at top
- Image paths broken → Use relative paths: `assets/images/file.png`

### Wiki pages not showing

**Refresh**:
- Wiki may cache for 5-10 minutes
- Try incognito/private browsing

**Permissions**:
- Verify repository is **Public** (not Private)
- Check Settings → Features → Wikis is enabled

### Private content leaked to GitHub

**IMMEDIATE ACTION**:
```bash
# Remove from history (use BFG Repo-Cleaner)
# https://rtyley.github.io/bfg-repo-cleaner/

# Or nuclear option (rewrite history)
git filter-branch --tree-filter 'rm -rf private' HEAD
git push --force
```

**Then**: Change all sensitive information (API keys, etc.)

---

## Support

🐛 **Setup issues**: [Open an issue](https://github.com/player2/signal_experiments/issues)
💬 **Questions**: [Discussions](https://github.com/player2/signal_experiments/discussions)

---

## Checklist

**Phase 1 Complete when:**
- [x] `private/` directory created and gitignored
- [x] `docs/` structure created
- [x] README.md updated
- [ ] GitHub Wiki enabled
- [ ] GitHub Pages enabled and deployed
- [ ] Wiki pages created (Home, Getting Started)
- [ ] Changes committed and pushed
- [ ] Website verified live
- [ ] Shared with community

**Ready for Phase 2:**
- [ ] Docker installed
- [ ] First Dockerfile created
- [ ] Experiment containerized successfully

---

**Next Steps**: Complete manual GitHub web interface steps, then begin Phase 2 Docker work.

**Estimated Time**: 30 minutes to complete Phase 1, then ready for 6-week Docker learning journey.
