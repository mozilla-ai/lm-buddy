from fastapi import FastAPI
from ray import serve

# from lm_buddy.serving.value import VALUE

app = FastAPI()


@serve.deployment
@serve.ingress(app)
class SimpleDeployment:
    @app.get("/value")
    def value(self, request):
        return {"value": 42}


deployment = SimpleDeployment.bind()
