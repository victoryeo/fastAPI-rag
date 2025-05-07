from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class QueryInput(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    context: Optional[str] = Field(None, max_length=5000)
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    
    @field_validator('question')
    @classmethod
    def question_must_be_appropriate(cls, v, info):
        # Basic profanity filter example
        forbidden_terms = ["inappropriate_term1", "inappropriate_term2"]
        if any(term in v.lower() for term in forbidden_terms):
            raise ValueError("Query contains inappropriate content")
        return v