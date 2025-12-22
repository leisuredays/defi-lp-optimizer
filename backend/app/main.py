"""
FastAPI Main Application

ML Optimization API for Uniswap V3 liquidity positions.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime

from app.config import settings
from app.api.v1 import optimize, health, predict, backtest

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(optimize.router, prefix="/api/v1", tags=["Optimization"])
app.include_router(predict.router, prefix="/api/v1", tags=["AI Prediction"])
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(backtest.router, prefix="/api/v1", tags=["Backtest"])


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup"""
    print(f"🚀 Starting {settings.API_TITLE} v{settings.API_VERSION}")
    print(f"📊 Model version: {settings.MODEL_VERSION}")
    print(f"🔑 Graph API configured: {'✓' if settings.GRAPH_API_KEY else '✗'}")
    if not settings.GRAPH_API_KEY:
        print("   ⚠️  Get your API key from: https://thegraph.com/studio/")


@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown"""
    print("👋 Shutting down ML Optimization API")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )
