"""
Main FastAPI application for the FXorcist dashboard.

Integrates all components and provides the main entry point.
"""

import logging
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from typing import List

from .routers import portfolio, market, system
from .auth import auth_service
from .cache import cache_instance, CacheConfig
from .websocket import connection_manager
from .middleware import setup_middleware
from .models import User

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    try:
        # Initialize services
        logger.info("Initializing services...")
        
        # Start cache service
        await cache_instance.start()
        logger.info("Cache service started")
        
        # Start WebSocket manager
        await connection_manager.start()
        logger.info("WebSocket manager started")
        
        yield
        
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down services...")
        
        # Stop WebSocket manager
        await connection_manager.stop()
        logger.info("WebSocket manager stopped")
        
        # Stop cache service
        await cache_instance.stop()
        logger.info("Cache service stopped")

def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="FXorcist Dashboard API",
        description="Trading dashboard backend API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    # Setup middleware
    setup_middleware(app, {
        "rate_limit": {
            "requests_per_minute": 60,
            "burst_limit": 100
        },
        "redis_url": "redis://localhost:6379/0"
    })
    
    # Include routers
    app.include_router(portfolio.router)
    app.include_router(market.router)
    app.include_router(system.router)
    
    # Add authentication endpoints
    @app.post("/token")
    async def login(username: str, password: str):
        """Login endpoint."""
        user = auth_service.authenticate_user(username, password)
        if not user:
            return {"error": "Invalid credentials"}
        
        access_token = auth_service.create_access_token(
            data={"sub": user.username, "scopes": user.scopes}
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer"
        }
    
    @app.get("/users/me")
    async def read_users_me(
        current_user: User = Depends(auth_service.get_current_active_user)
    ):
        """Get current user information."""
        return current_user
    
    # Add startup event handler
    @app.on_event("startup")
    async def startup_event():
        """Handle application startup."""
        logger.info("Starting FXorcist Dashboard API")
        
        # Initialize configuration
        config = {
            "environment": os.getenv("FXORCIST_ENV", "development"),
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0")
        }
        
        # Configure logging
        logging.getLogger().setLevel(config["log_level"])
        
        logger.info(f"Configuration loaded: {config}")
    
    # Add shutdown event handler
    @app.on_event("shutdown")
    async def shutdown_event():
        """Handle application shutdown."""
        logger.info("Shutting down FXorcist Dashboard API")
    
    # Add exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle uncaught exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if app.debug else None
            }
        )
    
    return app

def run_app():
    """Run the application."""
    app = create_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True,  # Enable auto-reload during development
        workers=4  # Number of worker processes
    )
    
    # Start server
    server = uvicorn.Server(config)
    server.run()

if __name__ == "__main__":
    run_app()