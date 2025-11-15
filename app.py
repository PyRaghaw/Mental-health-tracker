from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from textblob import TextBlob
import logging
import json
import re
from collections import deque
import asyncio

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="MindMate - AI Mental Health Platform",
    description="Production-grade Mental Health Support System with Advanced AI/ML",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# In-Memory Database (Replace with PostgreSQL/MongoDB in production)
users_db = {}
sessions_db = {}
conversations_db = {}

# ==================== AI/ML Models ====================

class AIModels:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        
    async def load_models(self):
        """Load all AI/ML models"""
        try:
            logger.info("Loading AI/ML models...")
            
            # 1. Mental Health Classification Model
            self.mental_health_tokenizer = AutoTokenizer.from_pretrained(
                "mental/mental-bert-base-uncased"
            )
            self.mental_health_model = AutoModelForSequenceClassification.from_pretrained(
                "mental/mental-bert-base-uncased"
            ).to(self.device)
            
            # 2. Emotion Detection
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 3. Sentiment Analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 4. Conversational AI (Therapy-focused)
            self.conversation_model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models_loaded = True
            logger.info("✅ All AI/ML models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Fallback to basic models
            self.models_loaded = False

ai_models = AIModels()

# ==================== Pydantic Models ====================

class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class MoodEntry(BaseModel):
    mood_score: int = Field(ge=1, le=10)
    emotions: List[str]
    notes: Optional[str] = None

class TherapySession(BaseModel):
    session_type: str  # "chat", "assessment", "crisis"
    initial_message: Optional[str] = None

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# ==================== Therapeutic Response Engine ====================

class TherapyEngine:
    """Advanced therapy response generator with context awareness"""
    
    def __init__(self):
        self.conversation_history = {}
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'self harm', 'hurt myself', 'no point living'
        ]
        self.therapy_techniques = {
            'cognitive': self._cognitive_reframing,
            'validation': self._emotional_validation,
            'exploration': self._exploratory_questions,
            'grounding': self._grounding_techniques,
            'coping': self._coping_strategies
        }
        
    def detect_crisis(self, message: str) -> bool:
        """Detect crisis situations"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.crisis_keywords)
    
    def analyze_emotional_state(self, message: str, emotion_scores: List[Dict]) -> Dict:
        """Deep emotional analysis"""
        # Get dominant emotion
        dominant_emotion = max(emotion_scores[0], key=lambda x: x['score'])
        
        # Sentiment analysis
        blob = TextBlob(message)
        sentiment_polarity = blob.sentiment.polarity
        
        # Emotional intensity
        intensity = dominant_emotion['score']
        
        return {
            'dominant_emotion': dominant_emotion['label'],
            'intensity': intensity,
            'sentiment': 'positive' if sentiment_polarity > 0 else 'negative' if sentiment_polarity < 0 else 'neutral',
            'polarity_score': sentiment_polarity,
            'all_emotions': emotion_scores[0]
        }
    
    def _cognitive_reframing(self, context: Dict) -> str:
        """Cognitive Behavioral Therapy technique"""
        responses = [
            "I hear that you're experiencing difficult thoughts. Let's explore this together - what evidence supports this thought, and what evidence might challenge it?",
            "Those thoughts sound really challenging. Can we look at this situation from a different angle? What might be another way to interpret what happened?",
            "It's understandable to think that way given your experience. What would you tell a close friend who shared similar thoughts with you?"
        ]
        return responses[hash(str(context)) % len(responses)]
    
    def _emotional_validation(self, context: Dict) -> str:
        """Validate and normalize emotions"""
        emotion = context.get('emotion', 'difficult')
        responses = [
            f"What you're feeling is completely valid. {emotion.capitalize()} emotions are a natural part of being human, and it's okay to experience them.",
            f"I really appreciate you sharing this with me. Feeling {emotion} in your situation makes a lot of sense.",
            f"Thank you for trusting me with these feelings. It takes courage to acknowledge and express {emotion} emotions."
        ]
        return responses[hash(str(context)) % len(responses)]
    
    def _exploratory_questions(self, context: Dict) -> str:
        """Ask therapeutic questions"""
        questions = [
            "Can you tell me more about when you first started noticing these feelings?",
            "What does this situation mean to you? What thoughts come up when you reflect on it?",
            "How has this been affecting your daily life - your sleep, relationships, or activities you usually enjoy?",
            "When you think about this, where do you feel it in your body? Sometimes our physical sensations can tell us a lot."
        ]
        return questions[hash(str(context)) % len(questions)]
    
    def _grounding_techniques(self, context: Dict) -> str:
        """Provide grounding exercises"""
        techniques = [
            "Let's try a quick grounding exercise together. Can you name 5 things you can see around you right now? This can help bring you back to the present moment.",
            "I notice you might be feeling overwhelmed. Try this: Take a deep breath in for 4 counts, hold for 4, and release for 4. Would you like to try that with me?",
            "When emotions feel intense, sometimes it helps to connect with your physical space. Try noticing the feeling of your feet on the ground or your body against the chair."
        ]
        return techniques[hash(str(context)) % len(techniques)]
    
    def _coping_strategies(self, context: Dict) -> str:
        """Suggest evidence-based coping strategies"""
        strategies = [
            "Based on what you've shared, some people find it helpful to: journal their thoughts, practice mindfulness, or engage in gentle physical activity. What resonates with you?",
            "It sounds like you could benefit from some self-care strategies. Have you tried any coping techniques before that helped you feel better?",
            "Building a coping toolbox can be really helpful. This might include things like calling a friend, listening to music, or spending time in nature. What activities usually help you feel more grounded?"
        ]
        return strategies[hash(str(context)) % len(strategies)]
    
    def generate_therapeutic_response(
        self, 
        message: str, 
        emotional_state: Dict,
        conversation_context: List[str]
    ) -> str:
        """Generate contextual therapeutic response"""
        
        # Crisis detection
        if self.detect_crisis(message):
            return self._crisis_response()
        
        # Determine appropriate therapy technique based on emotional state
        emotion = emotional_state['dominant_emotion']
        intensity = emotional_state['intensity']
        
        context = {
            'emotion': emotion,
            'intensity': intensity,
            'message': message,
            'history_length': len(conversation_context)
        }
        
        # High intensity negative emotions -> Validation + Grounding
        if intensity > 0.7 and emotional_state['sentiment'] == 'negative':
            validation = self._emotional_validation(context)
            grounding = self._grounding_techniques(context)
            return f"{validation}\n\n{grounding}"
        
        # Moderate emotions -> Exploration
        elif intensity > 0.4:
            if len(conversation_context) < 3:
                return self._exploratory_questions(context)
            else:
                return self._cognitive_reframing(context)
        
        # Lower intensity -> Coping strategies
        else:
            return self._coping_strategies(context)
    
    def _crisis_response(self) -> str:
        """Immediate crisis intervention response"""
        return """I'm really concerned about what you've shared with me. Your safety is the top priority right now.

🆘 **Immediate Help Available:**
- **National Suicide Prevention Lifeline (US):** 988 or 1-800-273-8255
- **Crisis Text Line:** Text HOME to 741741
- **International:** https://findahelpline.com

If you're in immediate danger, please:
1. Call emergency services (911 in US)
2. Go to your nearest emergency room
3. Contact a trusted friend or family member

I'm here with you, but I'm an AI and cannot provide emergency intervention. Please reach out to these professional resources immediately. Would you be willing to contact one of these services now?"""

therapy_engine = TherapyEngine()

# ==================== Authentication ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return users_db[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("🚀 Starting MindMate Backend...")
    await ai_models.load_models()
    logger.info("✅ Backend ready!")

@app.get("/")
async def root():
    return {
        "app": "MindMate - AI Mental Health Platform",
        "version": "2.0.0",
        "status": "operational",
        "models_loaded": ai_models.models_loaded
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "loaded": ai_models.models_loaded,
            "device": str(ai_models.device)
        }
    }

@app.post("/api/auth/register")
async def register(user: UserRegister):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = get_password_hash(user.password)
    user_data = {
        "user_id": f"user_{len(users_db) + 1}",
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.now(),
        "conversations": [],
        "mood_history": [],
        "assessment_scores": {}
    }
    users_db[user.username] = user_data
    
    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**user_data)
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_data = users_db[user.username]
    if not verify_password(user.password, user_data["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**user_data)
    }

@app.post("/api/chat")
async def chat(message: ChatMessage, current_user: dict = Depends(get_current_user)):
    """Main therapy chat endpoint with advanced AI/ML"""
    
    if not ai_models.models_loaded:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    try:
        user_message = message.message
        session_id = message.session_id or f"session_{datetime.now().timestamp()}"
        
        # Get or create conversation history
        if session_id not in conversations_db:
            conversations_db[session_id] = {
                "user_id": current_user["user_id"],
                "started_at": datetime.now(),
                "messages": []
            }
        
        conversation = conversations_db[session_id]
        
        # Emotion Detection
        emotion_scores = ai_models.emotion_classifier(user_message)
        
        # Emotional State Analysis
        emotional_state = therapy_engine.analyze_emotional_state(
            user_message, 
            emotion_scores
        )
        
        # Generate Therapeutic Response
        conversation_context = [msg["content"] for msg in conversation["messages"][-5:]]
        
        ai_response = therapy_engine.generate_therapeutic_response(
            user_message,
            emotional_state,
            conversation_context
        )
        
        # Save conversation
        conversation["messages"].extend([
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now(),
                "emotional_state": emotional_state
            },
            {
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now()
            }
        ])
        
        return {
            "response": ai_response,
            "session_id": session_id,
            "emotional_analysis": emotional_state,
            "conversation_length": len(conversation["messages"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/api/mood/track")
async def track_mood(mood: MoodEntry, current_user: dict = Depends(get_current_user)):
    """Track daily mood with ML analysis"""
    
    mood_entry = {
        "mood_score": mood.mood_score,
        "emotions": mood.emotions,
        "notes": mood.notes,
        "timestamp": datetime.now(),
        "analysis": {
            "trend": "stable",  # Calculate from history
            "risk_level": "low" if mood.mood_score > 5 else "medium"
        }
    }
    
    current_user["mood_history"].append(mood_entry)
    
    return {
        "status": "success",
        "mood_entry": mood_entry,
        "weekly_average": sum([m["mood_score"] for m in current_user["mood_history"][-7:]]) / min(len(current_user["mood_history"]), 7)
    }

@app.get("/api/user/dashboard")
async def get_dashboard(current_user: dict = Depends(get_current_user)):
    """Get user dashboard with insights"""
    
    mood_history = current_user.get("mood_history", [])
    conversations = current_user.get("conversations", [])
    
    # Calculate insights
    avg_mood = sum([m["mood_score"] for m in mood_history[-30:]]) / len(mood_history) if mood_history else 5
    
    return {
        "user": UserResponse(**current_user),
        "stats": {
            "total_sessions": len(conversations),
            "mood_entries": len(mood_history),
            "average_mood_30d": round(avg_mood, 1),
            "streak_days": len(mood_history)  # Simplified
        },
        "recent_moods": mood_history[-7:],
        "insights": [
            "You've been consistently tracking your mood - great job!",
            f"Your average mood over the last 30 days is {round(avg_mood, 1)}/10",
            "Consider setting aside 10 minutes daily for self-reflection"
        ]
    }

@app.get("/api/conversations/history")
async def get_conversation_history(current_user: dict = Depends(get_current_user)):
    """Get user's conversation history"""
    
    user_conversations = {
        session_id: conv 
        for session_id, conv in conversations_db.items() 
        if conv["user_id"] == current_user["user_id"]
    }
    
    return {
        "conversations": user_conversations,
        "total": len(user_conversations)
    }

@app.get("/api/resources")
async def get_resources():
    """Mental health resources and helplines"""
    return {
        "emergency": {
            "suicide_prevention": {
                "us": "988 or 1-800-273-8255",
                "text": "Text HOME to 741741",
                "international": "https://findahelpline.com"
            },
            "emergency_services": "911 (US) or local emergency number"
        },
        "support": {
            "therapy": "BetterHelp, Talkspace, Psychology Today",
            "support_groups": "NAMI, Mental Health America",
            "meditation": "Headspace, Calm, Insight Timer"
        },
        "educational": [
            {
                "title": "Understanding Anxiety",
                "url": "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
            },
            {
                "title": "Depression Information",
                "url": "https://www.nimh.nih.gov/health/topics/depression"
            }
        ]
    }

# ==================== Admin Endpoints ====================

@app.get("/api/admin/stats")
async def get_admin_stats():
    """System statistics (add proper admin auth in production)"""
    return {
        "total_users": len(users_db),
        "total_conversations": len(conversations_db),
        "total_messages": sum(len(conv["messages"]) for conv in conversations_db.values()),
        "models_loaded": ai_models.models_loaded,
        "system_health": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from textblob import TextBlob
import logging
import json
import re
from collections import deque
import asyncio

# Configuration
SECRET_KEY = "your-secret-key-change-this-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="MindMate - AI Mental Health Platform",
    description="Production-grade Mental Health Support System with Advanced AI/ML",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# In-Memory Database (Replace with PostgreSQL/MongoDB in production)
users_db = {}
sessions_db = {}
conversations_db = {}

# ==================== AI/ML Models ====================

class AIModels:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        
    async def load_models(self):
        """Load all AI/ML models"""
        try:
            logger.info("Loading AI/ML models...")
            
            # 1. Mental Health Classification Model
            self.mental_health_tokenizer = AutoTokenizer.from_pretrained(
                "mental/mental-bert-base-uncased"
            )
            self.mental_health_model = AutoModelForSequenceClassification.from_pretrained(
                "mental/mental-bert-base-uncased"
            ).to(self.device)
            
            # 2. Emotion Detection
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True,
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 3. Sentiment Analysis
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # 4. Conversational AI (Therapy-focused)
            self.conversation_model = pipeline(
                "text-generation",
                model="microsoft/DialoGPT-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models_loaded = True
            logger.info("✅ All AI/ML models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            # Fallback to basic models
            self.models_loaded = False

ai_models = AIModels()

# ==================== Pydantic Models ====================

class UserRegister(BaseModel):
    username: str
    email: str
    password: str
    full_name: Optional[str] = None

class UserLogin(BaseModel):
    username: str
    password: str

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class MoodEntry(BaseModel):
    mood_score: int = Field(ge=1, le=10)
    emotions: List[str]
    notes: Optional[str] = None

class TherapySession(BaseModel):
    session_type: str  # "chat", "assessment", "crisis"
    initial_message: Optional[str] = None

class UserResponse(BaseModel):
    user_id: str
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)

# ==================== Therapeutic Response Engine ====================

class TherapyEngine:
    """Advanced therapy response generator with context awareness"""
    
    def __init__(self):
        self.conversation_history = {}
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end my life', 'want to die',
            'self harm', 'hurt myself', 'no point living'
        ]
        self.therapy_techniques = {
            'cognitive': self._cognitive_reframing,
            'validation': self._emotional_validation,
            'exploration': self._exploratory_questions,
            'grounding': self._grounding_techniques,
            'coping': self._coping_strategies
        }
        
    def detect_crisis(self, message: str) -> bool:
        """Detect crisis situations"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.crisis_keywords)
    
    def analyze_emotional_state(self, message: str, emotion_scores: List[Dict]) -> Dict:
        """Deep emotional analysis"""
        # Get dominant emotion
        dominant_emotion = max(emotion_scores[0], key=lambda x: x['score'])
        
        # Sentiment analysis
        blob = TextBlob(message)
        sentiment_polarity = blob.sentiment.polarity
        
        # Emotional intensity
        intensity = dominant_emotion['score']
        
        return {
            'dominant_emotion': dominant_emotion['label'],
            'intensity': intensity,
            'sentiment': 'positive' if sentiment_polarity > 0 else 'negative' if sentiment_polarity < 0 else 'neutral',
            'polarity_score': sentiment_polarity,
            'all_emotions': emotion_scores[0]
        }
    
    def _cognitive_reframing(self, context: Dict) -> str:
        """Cognitive Behavioral Therapy technique"""
        responses = [
            "I hear that you're experiencing difficult thoughts. Let's explore this together - what evidence supports this thought, and what evidence might challenge it?",
            "Those thoughts sound really challenging. Can we look at this situation from a different angle? What might be another way to interpret what happened?",
            "It's understandable to think that way given your experience. What would you tell a close friend who shared similar thoughts with you?"
        ]
        return responses[hash(str(context)) % len(responses)]
    
    def _emotional_validation(self, context: Dict) -> str:
        """Validate and normalize emotions"""
        emotion = context.get('emotion', 'difficult')
        responses = [
            f"What you're feeling is completely valid. {emotion.capitalize()} emotions are a natural part of being human, and it's okay to experience them.",
            f"I really appreciate you sharing this with me. Feeling {emotion} in your situation makes a lot of sense.",
            f"Thank you for trusting me with these feelings. It takes courage to acknowledge and express {emotion} emotions."
        ]
        return responses[hash(str(context)) % len(responses)]
    
    def _exploratory_questions(self, context: Dict) -> str:
        """Ask therapeutic questions"""
        questions = [
            "Can you tell me more about when you first started noticing these feelings?",
            "What does this situation mean to you? What thoughts come up when you reflect on it?",
            "How has this been affecting your daily life - your sleep, relationships, or activities you usually enjoy?",
            "When you think about this, where do you feel it in your body? Sometimes our physical sensations can tell us a lot."
        ]
        return questions[hash(str(context)) % len(questions)]
    
    def _grounding_techniques(self, context: Dict) -> str:
        """Provide grounding exercises"""
        techniques = [
            "Let's try a quick grounding exercise together. Can you name 5 things you can see around you right now? This can help bring you back to the present moment.",
            "I notice you might be feeling overwhelmed. Try this: Take a deep breath in for 4 counts, hold for 4, and release for 4. Would you like to try that with me?",
            "When emotions feel intense, sometimes it helps to connect with your physical space. Try noticing the feeling of your feet on the ground or your body against the chair."
        ]
        return techniques[hash(str(context)) % len(techniques)]
    
    def _coping_strategies(self, context: Dict) -> str:
        """Suggest evidence-based coping strategies"""
        strategies = [
            "Based on what you've shared, some people find it helpful to: journal their thoughts, practice mindfulness, or engage in gentle physical activity. What resonates with you?",
            "It sounds like you could benefit from some self-care strategies. Have you tried any coping techniques before that helped you feel better?",
            "Building a coping toolbox can be really helpful. This might include things like calling a friend, listening to music, or spending time in nature. What activities usually help you feel more grounded?"
        ]
        return strategies[hash(str(context)) % len(strategies)]
    
    def generate_therapeutic_response(
        self, 
        message: str, 
        emotional_state: Dict,
        conversation_context: List[str]
    ) -> str:
        """Generate contextual therapeutic response"""
        
        # Crisis detection
        if self.detect_crisis(message):
            return self._crisis_response()
        
        # Determine appropriate therapy technique based on emotional state
        emotion = emotional_state['dominant_emotion']
        intensity = emotional_state['intensity']
        
        context = {
            'emotion': emotion,
            'intensity': intensity,
            'message': message,
            'history_length': len(conversation_context)
        }
        
        # High intensity negative emotions -> Validation + Grounding
        if intensity > 0.7 and emotional_state['sentiment'] == 'negative':
            validation = self._emotional_validation(context)
            grounding = self._grounding_techniques(context)
            return f"{validation}\n\n{grounding}"
        
        # Moderate emotions -> Exploration
        elif intensity > 0.4:
            if len(conversation_context) < 3:
                return self._exploratory_questions(context)
            else:
                return self._cognitive_reframing(context)
        
        # Lower intensity -> Coping strategies
        else:
            return self._coping_strategies(context)
    
    def _crisis_response(self) -> str:
        """Immediate crisis intervention response"""
        return """I'm really concerned about what you've shared with me. Your safety is the top priority right now.

🆘 **Immediate Help Available:**
- **National Suicide Prevention Lifeline (US):** 988 or 1-800-273-8255
- **Crisis Text Line:** Text HOME to 741741
- **International:** https://findahelpline.com

If you're in immediate danger, please:
1. Call emergency services (911 in US)
2. Go to your nearest emergency room
3. Contact a trusted friend or family member

I'm here with you, but I'm an AI and cannot provide emergency intervention. Please reach out to these professional resources immediately. Would you be willing to contact one of these services now?"""

therapy_engine = TherapyEngine()

# ==================== Authentication ====================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return users_db[username]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

# ==================== API Endpoints ====================

@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("🚀 Starting MindMate Backend...")
    await ai_models.load_models()
    logger.info("✅ Backend ready!")

@app.get("/")
async def root():
    return {
        "app": "MindMate - AI Mental Health Platform",
        "version": "2.0.0",
        "status": "operational",
        "models_loaded": ai_models.models_loaded
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "loaded": ai_models.models_loaded,
            "device": str(ai_models.device)
        }
    }

@app.post("/api/auth/register")
async def register(user: UserRegister):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed_password = get_password_hash(user.password)
    user_data = {
        "user_id": f"user_{len(users_db) + 1}",
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "created_at": datetime.now(),
        "conversations": [],
        "mood_history": [],
        "assessment_scores": {}
    }
    users_db[user.username] = user_data
    
    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**user_data)
    }

@app.post("/api/auth/login")
async def login(user: UserLogin):
    if user.username not in users_db:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    user_data = users_db[user.username]
    if not verify_password(user.password, user_data["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": UserResponse(**user_data)
    }

@app.post("/api/chat")
async def chat(message: ChatMessage, current_user: dict = Depends(get_current_user)):
    """Main therapy chat endpoint with advanced AI/ML"""
    
    if not ai_models.models_loaded:
        raise HTTPException(status_code=503, detail="AI models not loaded")
    
    try:
        user_message = message.message
        session_id = message.session_id or f"session_{datetime.now().timestamp()}"
        
        # Get or create conversation history
        if session_id not in conversations_db:
            conversations_db[session_id] = {
                "user_id": current_user["user_id"],
                "started_at": datetime.now(),
                "messages": []
            }
        
        conversation = conversations_db[session_id]
        
        # Emotion Detection
        emotion_scores = ai_models.emotion_classifier(user_message)
        
        # Emotional State Analysis
        emotional_state = therapy_engine.analyze_emotional_state(
            user_message, 
            emotion_scores
        )
        
        # Generate Therapeutic Response
        conversation_context = [msg["content"] for msg in conversation["messages"][-5:]]
        
        ai_response = therapy_engine.generate_therapeutic_response(
            user_message,
            emotional_state,
            conversation_context
        )
        
        # Save conversation
        conversation["messages"].extend([
            {
                "role": "user",
                "content": user_message,
                "timestamp": datetime.now(),
                "emotional_state": emotional_state
            },
            {
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now()
            }
        ])
        
        return {
            "response": ai_response,
            "session_id": session_id,
            "emotional_analysis": emotional_state,
            "conversation_length": len(conversation["messages"]),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.post("/api/mood/track")
async def track_mood(mood: MoodEntry, current_user: dict = Depends(get_current_user)):
    """Track daily mood with ML analysis"""
    
    mood_entry = {
        "mood_score": mood.mood_score,
        "emotions": mood.emotions,
        "notes": mood.notes,
        "timestamp": datetime.now(),
        "analysis": {
            "trend": "stable",  # Calculate from history
            "risk_level": "low" if mood.mood_score > 5 else "medium"
        }
    }
    
    current_user["mood_history"].append(mood_entry)
    
    return {
        "status": "success",
        "mood_entry": mood_entry,
        "weekly_average": sum([m["mood_score"] for m in current_user["mood_history"][-7:]]) / min(len(current_user["mood_history"]), 7)
    }

@app.get("/api/user/dashboard")
async def get_dashboard(current_user: dict = Depends(get_current_user)):
    """Get user dashboard with insights"""
    
    mood_history = current_user.get("mood_history", [])
    conversations = current_user.get("conversations", [])
    
    # Calculate insights
    avg_mood = sum([m["mood_score"] for m in mood_history[-30:]]) / len(mood_history) if mood_history else 5
    
    return {
        "user": UserResponse(**current_user),
        "stats": {
            "total_sessions": len(conversations),
            "mood_entries": len(mood_history),
            "average_mood_30d": round(avg_mood, 1),
            "streak_days": len(mood_history)  # Simplified
        },
        "recent_moods": mood_history[-7:],
        "insights": [
            "You've been consistently tracking your mood - great job!",
            f"Your average mood over the last 30 days is {round(avg_mood, 1)}/10",
            "Consider setting aside 10 minutes daily for self-reflection"
        ]
    }

@app.get("/api/conversations/history")
async def get_conversation_history(current_user: dict = Depends(get_current_user)):
    """Get user's conversation history"""
    
    user_conversations = {
        session_id: conv 
        for session_id, conv in conversations_db.items() 
        if conv["user_id"] == current_user["user_id"]
    }
    
    return {
        "conversations": user_conversations,
        "total": len(user_conversations)
    }

@app.get("/api/resources")
async def get_resources():
    """Mental health resources and helplines"""
    return {
        "emergency": {
            "suicide_prevention": {
                "us": "988 or 1-800-273-8255",
                "text": "Text HOME to 741741",
                "international": "https://findahelpline.com"
            },
            "emergency_services": "911 (US) or local emergency number"
        },
        "support": {
            "therapy": "BetterHelp, Talkspace, Psychology Today",
            "support_groups": "NAMI, Mental Health America",
            "meditation": "Headspace, Calm, Insight Timer"
        },
        "educational": [
            {
                "title": "Understanding Anxiety",
                "url": "https://www.nimh.nih.gov/health/topics/anxiety-disorders"
            },
            {
                "title": "Depression Information",
                "url": "https://www.nimh.nih.gov/health/topics/depression"
            }
        ]
    }


@app.get("/api/admin/stats")
async def get_admin_stats():
    """System statistics (add proper admin auth in production)"""
    return {
        "total_users": len(users_db),
        "total_conversations": len(conversations_db),
        "total_messages": sum(len(conv["messages"]) for conv in conversations_db.values()),
        "models_loaded": ai_models.models_loaded,
        "system_health": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
