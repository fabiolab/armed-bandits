from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# Ajout du middleware CORS à votre application FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/bandits/rewards")
def get_bandits_rewards():
    # Exemple de données de récompense
    rewards_data = [
        {"bandit_id": 0, "mean_reward": 2.45, "date": "2023-06-19T14:32:09Z"},
        {"bandit_id": 0, "mean_reward": 2.1, "date": "2023-06-19T14:00:09Z"},
        {"bandit_id": 2, "mean_reward": 1.1, "date": "2023-06-19T14:32:09Z"},
        {"bandit_id": 2, "mean_reward": 1.9, "date": "2023-06-19T14:00:09Z"},
    ]

    # Renvoie les données au format JSON
    return JSONResponse(content=rewards_data)


# Exécution de l'API avec uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
