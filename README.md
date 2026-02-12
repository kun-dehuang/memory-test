# Memory Solution Evaluation Tool

A full-stack multi-modal RAG application for comparing Mem0 and Zep memory providers with Google Gemini 1.5 Flash vision analysis.

## Architecture

- **Backend**: FastAPI deployed on Railway
- **Frontend**: React + Vite + Tailwind CSS deployed on GitHub Pages
- **Vision AI**: Google Gemini 1.5 Flash
- **Memory Providers**: Mem0 and Zep Cloud for semantic storage/retrieval

## Core Features

### 1. Identity Anchor (身份锚点)
- System extracts protagonist appearance features from `identity/self.jpg`
- Features stored globally and used for protagonist detection in all photos
- Supports hair, glasses, body type, clothing style recognition

### 2. Visual Anthropologist Pipeline
- Custom prompt template via `GEMINI_ANALYSIS_PROMPT` environment variable
- Default analysis includes:
  - 0. Protagonist recognition with `【主角】` tag
  - 1. Scene environment description
  - 2. Human interaction analysis
  - 3. Object/consumption identification
  - 4. Action/activity detection
  - 5. Visual style assessment

### 3. Dual Storage Strategy
- **Mem0**: Stores rich text content with protagonist tags for entity learning
- **Zep Cloud**: Stores scene description with EXIF/GPS metadata

### 4. Smart Query Processing
- Automatically replaces "我", "我的", "我也要", "我在" with "【主角】"
- Enables natural self-reference queries

## Project Structure

```
.
├── backend/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   └── memory_service.py  # Core memory logic with Gemini integration
│   ├── main.py                # FastAPI application
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Railway deployment
│   └── .env.example           # Environment variables template
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── ResultCard.jsx   # Result card with protagonist badge
│   │   └── App.jsx
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
├── identity/
│   └── self.jpg              # Protagonist reference photo
└── photos/                       # Upload your images here
```

## Setup & Deployment

### 1. Environment Variables

Copy `backend/.env.example` to `backend/.env` and configure:

```bash
# Zep Cloud Configuration (Required)
ZEP_API_KEY=your_zep_api_key_here

# Gemini Configuration (Required for image analysis)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Optional: Custom Visual Anthropologist Prompt
# GEMINI_ANALYSIS_PROMPT=Your custom prompt here...

# Mem0 Configuration
MEM0_API_KEY=your_mem0_api_key_here
MEM0_API_BASE=https://api.mem0.ai
MEM0_USER_ID=memory-test-user

# App Configuration
PHOTOS_DIR=photos
IDENTITY_DIR=identity
PROTAGONIST_TAG=【主角】

# CORS Configuration
FRONTEND_URL=https://your-username.github.io
```

### 2. Backend (Railway)

1. Create a new Railway project from this repository
2. Set Root Directory to `backend` in Railway project settings
3. Set all environment variables in Railway dashboard
4. Deploy!

### 3. Frontend (GitHub Pages)

1. Install dependencies: `cd frontend && npm install`
2. Build app: `npm run build`
3. Deploy `dist/` folder to GitHub Pages

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/scan` | POST | Scan photos with Gemini analysis and store in memory |
| `/api/reset` | DELETE | Clear all memories from both providers |
| `/api/search` | POST | Search with `{ "query": "...", "provider": "mem0\|zep" }` |
| `/static/*` | GET | Access uploaded photos |
| `/identity/*` | GET | Access identity photos |

## Usage

1. Place protagonist reference photo in `identity/self.jpg`
2. Add photos to analyze in `photos/` directory
3. Click "Re-scan Photos" to:
   - Extract protagonist features from `identity/self.jpg`
   - Analyze each photo concurrently with Gemini
   - Store in both Mem0 and Zep
4. Search using natural language:
   - "我在咖啡馆的照片" (Photos of me at cafe)
   - "有主角的户外照片" (Outdoor photos with protagonist)
5. Compare results between Mem0 and Zep

## Technical Details

### Image Formats Supported
- JPEG, PNG, WebP, GIF, BMP, TIFF
- HEIC/HEIF (via pillow-heif)

### Concurrent Processing
- Photos are analyzed concurrently using `asyncio.gather`
- Error handling ensures single photo failure doesn't stop the batch

### Protagonist Detection Flow
1. System startup: Check `identity/self.jpg`
2. Extract appearance features using Gemini
3. For each photo: Include features in analysis prompt
4. Tag photos containing protagonist with `【主角】`

### Search Query Processing
- User query: "我在咖啡馆的照片"
- Preprocessed: "【主角】在咖啡馆的照片"
- Enables semantic matching against tagged memories

## Key Dependencies

### Backend
- `zep-cloud` - Zep Cloud Python SDK
- `mem0ai` - Mem0 AI Memory
- `google-generativeai` - Gemini Vision API
- `pillow` + `pillow-heif` - Image processing
- `pydantic` + `pydantic-settings` - Settings management

### Sources
- [Zep Cloud Documentation](https://help.getzep.com/)
- [Zep Cloud GitHub](https://github.com/getzep/zep-cloud)
- [zep-cloud PyPI](https://pypi.org/project/zep-cloud/)
