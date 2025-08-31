# Gather Deployment Guide

## Quick Deploy Options

### 1. Railway (Easiest - Recommended)

**Step 1**: Build frontend
```bash
cd web-app
npm install
npm run build
```

**Step 2**: Deploy to Railway
1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway will auto-detect and deploy using the `railway.json` config
4. Set environment variables in Railway dashboard:
   - `FLASK_ENV=production`
   - `PORT=5001`

**Cost**: Free tier includes 500 hours/month

---

### 2. Vercel Frontend + Railway Backend (Good Separation)

**Frontend (Vercel)**:
1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repo
3. Set build command: `cd web-app && npm run build`
4. Set output directory: `web-app/dist`

**Backend (Railway)**:
- Follow Railway steps above
- Update frontend API URL to Railway backend URL

---

### 3. Heroku (Classic Choice)

**Step 1**: Install Heroku CLI
```bash
# macOS
brew install heroku/brew/heroku
```

**Step 2**: Deploy
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set buildpack for Python
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

---

### 4. DigitalOcean App Platform (More Control)

1. Go to [DigitalOcean](https://cloud.digitalocean.com/apps)
2. Create new app from GitHub
3. Configure:
   - **Backend**: Python app, auto-detect from `requirements.txt`
   - **Frontend**: Static site, build with `npm run build`

---

## Production Checklist

Before deploying, make sure:

- [ ] Frontend is built (`npm run build`)
- [ ] Model file is committed to repo (check `models/current_model.keras`)
- [ ] Environment variables are set
- [ ] CORS is configured for production domain
- [ ] Error handling is robust

## Environment Variables Needed

```bash
FLASK_ENV=production
PORT=5001
HOST=0.0.0.0
```

## Model Size Consideration

Your TensorFlow model (~87MB) might hit some deployment limits:
- **Railway**: 1GB limit (✅ should work)
- **Heroku**: 500MB slug size (⚠️ might be tight)
- **Vercel**: Not suitable for backend with large models

**Recommendation**: Start with Railway, it's designed for this type of application.
