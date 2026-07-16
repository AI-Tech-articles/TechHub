# TechHub Hugo Rebuild

## What was migrated from Jekyll

All layout bugs fixed and rebuilt as a Hugo site with custom `techhub` theme.

## File Structure

```
.
├── hugo.toml
├── content/
│   └── posts/
│       ├── 2026-07-15-ai-tech-2026-06-30.md
│       ├── 2026-07-15-asynchronous-sequence-batching.md
│       ├── 2026-07-15-vllm-vs-sglang.md
│       ├── 2026-07-15-vllm-pagedattention-debugging.md
│       ├── 2026-07-15-multi-node-pipeline-parallelism.md
│       └── 2026-07-15-tensor-parallelism-coordination.md
├── themes/
│   └── techhub/
│       ├── theme.toml
│       ├── layouts/
│       │   ├── _default/
│       │   │   ├── baseof.html
│       │   │   ├── list.html
│       │   │   └── single.html
│       │   └── partials/
│       │       ├── head.html
│       │       ├── header.html
│       │       ├── footer.html
│       │       └── ad-slot.html
│       └── assets/
│           └── css/
│               └── style.scss
└── static/
    └── images/
        └── default-og.png   (you need to add this)
```

## Build Instructions

### 1. Install Hugo

```bash
# macOS
brew install hugo

# Linux (Debian/Ubuntu)
sudo apt install hugo

# Or download from https://github.com/gohugoio/hugo/releases
```

### 2. Build the site

```bash
cd hugo-techhub
hugo server -D          # Development server with drafts
hugo                    # Build to public/ directory
```

### 3. Deploy to GitHub Pages

```bash
# Build with baseURL pointing to your GitHub Pages domain
hugo --baseURL "https://ai-tech-articles.github.io/TechHub/"

# The output is in public/ — push this to your gh-pages branch
# Or use GitHub Actions (see .github/workflows/hugo.yml below)
```

## GitHub Actions Workflow

Create `.github/workflows/hugo.yml`:

```yaml
name: Deploy Hugo to GitHub Pages

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          submodules: recursive
          fetch-depth: 0
      - name: Setup Hugo
        uses: peaceiris/actions-hugo@v2
        with:
          hugo-version: '0.124.0'
          extended: true
      - name: Build
        run: hugo --minify --baseURL "https://ai-tech-articles.github.io/TechHub/"
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public

  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

## Bug Fixes Applied (same as Jekyll fix pack)

1. ✅ **Empty CSS** → Full SCSS stylesheet in `themes/techhub/assets/css/style.scss`
2. ✅ **Ad placeholders as visible text** → Hidden `<div>` partials (`ad-slot.html`)
3. ✅ **Duplicate titles** → Title rendered exactly once in `<h2>`
4. ✅ **Transmission IDs leaked** → `data-transmission-id` attribute only
5. ✅ **Mermaid diagrams not rendering** → Mermaid v10 ES module in `baseof.html`
6. ✅ **Missing header/footer/nav** → Sticky header + footer in partials
7. ✅ **Feed.xml CDATA leak** → Hugo's built-in RSS template (no CDATA issues)
8. ✅ **SEO meta tags** → Open Graph, Twitter Cards, JSON-LD structured data
9. ✅ **404 page** → Hugo's default 404 template styled with theme
10. ✅ **Responsive mobile layout** → SCSS `@media` queries
11. ✅ **Post schema markup** → `itemscope itemtype="TechArticle"` + `itemprop`
12. ✅ **Code blocks styled** → `pre`, `code`, `blockquote`, `table` all styled

## Manual steps YOU must do

1. **Add your OpenGraph image** to `static/images/default-og.png`
2. **Add remaining articles** — copy your other `_posts/*.md` files into `content/posts/`, converting front matter from Jekyll to Hugo format (change `layout: post` to `draft: false`, etc.)
3. **Delete duplicate posts** if any exist in `content/posts/`
4. **Commit & push** — GitHub Actions will build and deploy automatically
