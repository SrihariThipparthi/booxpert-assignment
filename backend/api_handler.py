import logging

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.name_matching import NameMatcher
from services.recipe_bot import RecipeBot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Name Matching & Recipe Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

name_matcher = NameMatcher()
recipe_bot = RecipeBot()


class NameRequest(BaseModel):
    name: str


class RecipeRequest(BaseModel):
    ingredients: str


class NameMatch(BaseModel):
    name: str
    score: float


class NameMatchResponse(BaseModel):
    input_name: str
    best_match: NameMatch
    all_matches: list[NameMatch]


class RecipeResponse(BaseModel):
    recipe: str
    generated_by: str


@app.get("/")
async def root():
    return {
        "message": "Name Matching & Recipe Chatbot API",
        "endpoints": {
            "health": "/health",
            "match_names": "/api/match-names",
            "get_recipe": "/api/get-recipe",
        },
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "api"}


@app.post("/api/match-names", response_model=NameMatchResponse)
async def match_names(request: NameRequest):
    try:
        if not request.name or not request.name.strip():
            raise HTTPException(status_code=400, detail="Name cannot be empty")

        results = name_matcher.find_similar_names(request.name.strip())

        return NameMatchResponse(
            input_name=request.name,
            best_match=NameMatch(
                name=results["best_match"]["name"], score=results["best_match"]["score"]
            ),
            all_matches=[
                NameMatch(name=match["name"], score=match["score"])
                for match in results["all_matches"]
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/get-recipe", response_model=RecipeResponse)
async def get_recipe(request: RecipeRequest):
    try:
        if not request.ingredients or not request.ingredients.strip():
            raise HTTPException(status_code=400, detail="Ingredients cannot be empty")

        result = recipe_bot.generate_recipe(request.ingredients.strip())

        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])

        return RecipeResponse(recipe=result["recipe"], generated_by="recipe-bot-lora")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting Name Matching & Recipe Chatbot API Server")
    logger.info("API will be available at: http://127.0.0.1:8000")
    logger.info("API Documentation: http://127.0.0.1:8000/docs")

    uvicorn.run(
        app, host="0.0.0.0", port=8000, timeout_keep_alive=300, log_level="info"
    )
