"""
Authentication and authorization system for the dashboard API.

Implements JWT-based authentication with role-based access control.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, List
import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from passlib.context import CryptContext
from pydantic import BaseModel, ValidationError
import logging
from enum import Enum

# Setup logging
logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        "read:portfolio": "Read portfolio information",
        "write:trades": "Execute trades",
        "read:system": "Read system status",
        "admin": "Full system access"
    }
)

class Role(str, Enum):
    """User roles for access control."""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"

class User(BaseModel):
    """User model with authentication information."""
    username: str
    email: str
    full_name: Optional[str]
    disabled: bool = False
    role: Role
    scopes: List[str]

class Token(BaseModel):
    """JWT token model."""
    access_token: str
    token_type: str
    expires_in: int
    scope: str

class TokenData(BaseModel):
    """JWT token payload data."""
    username: str
    scopes: List[str]
    role: Role
    exp: datetime

class AuthConfig:
    """Authentication configuration."""
    SECRET_KEY = "your-secret-key"  # Should be loaded from environment
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30

class AuthService:
    """Authentication service implementing security features."""

    def __init__(self):
        """Initialize authentication service."""
        self.users_db: Dict[str, Dict] = {}  # In-memory user store for demo
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        return pwd_context.hash(password)
    
    def get_user(self, username: str) -> Optional[User]:
        """Get user by username."""
        if username in self.users_db:
            user_dict = self.users_db[username]
            return User(**user_dict)
        return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user = self.get_user(username)
        if not user:
            return None
        if not self.verify_password(password, self.users_db[username]["hashed_password"]):
            return None
        return user
    
    def create_access_token(
        self,
        data: dict,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        
        to_encode.update({"exp": expire})
        
        try:
            encoded_jwt = jwt.encode(
                to_encode,
                AuthConfig.SECRET_KEY,
                algorithm=AuthConfig.ALGORITHM
            )
            return encoded_jwt
        except Exception as e:
            logger.error(f"Error creating access token: {e}")
            raise HTTPException(
                status_code=500,
                detail="Could not create access token"
            )
    
    async def get_current_user(
        self,
        security_scopes: SecurityScopes,
        token: str = Depends(oauth2_scheme)
    ) -> User:
        """Get current user from JWT token."""
        if security_scopes.scopes:
            authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
        else:
            authenticate_value = "Bearer"
        
        credentials_exception = HTTPException(
            status_code=401,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": authenticate_value},
        )
        
        try:
            payload = jwt.decode(
                token,
                AuthConfig.SECRET_KEY,
                algorithms=[AuthConfig.ALGORITHM]
            )
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            
            token_scopes = payload.get("scopes", [])
            token_data = TokenData(
                username=username,
                scopes=token_scopes,
                role=payload.get("role"),
                exp=payload.get("exp")
            )
        except (jwt.PyJWTError, ValidationError) as e:
            logger.error(f"Token validation error: {e}")
            raise credentials_exception
        
        user = self.get_user(username=token_data.username)
        if user is None:
            raise credentials_exception
        
        # Verify required scopes
        for scope in security_scopes.scopes:
            if scope not in token_data.scopes:
                raise HTTPException(
                    status_code=403,
                    detail="Not enough permissions",
                    headers={"WWW-Authenticate": authenticate_value},
                )
        
        return user
    
    async def get_current_active_user(
        self,
        current_user: User = Security(get_current_user, scopes=[])
    ) -> User:
        """Get current active user."""
        if current_user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user

    def has_permission(self, user: User, required_scopes: List[str]) -> bool:
        """Check if user has required permissions."""
        if user.role == Role.ADMIN:
            return True
        return all(scope in user.scopes for scope in required_scopes)

# Global auth service instance
auth_service = AuthService()