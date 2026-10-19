from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier
from dotenv import load_dotenv
import os
load_dotenv()

classifier = TypeSafeClassifier(api_key= os.environ.get("JEV_API_KEY"))

response = classifier.invoke(
    {
        "state": (
            "The deploy failed twice and customers are seeing 500s. "
            "Can someone look now?"
        ),
        "questions": {
            "urgent": Noul(instructions="Does this need attention right now?"),
            "team": Choice(
                instructions="Which team should pick this up?",
                criteria={
                    "infra": "Deploys, availability, and on-call incidents.",
                    "billing": "Payments, invoices, and subscriptions.",
                },
            ),
            "severity": Score(
                instructions="How severe is the impact?",
                criteria=["Cosmetic.", "Degraded for some users.", "Full outage."],
            ),
        },
    }
)

print(response.nouls["urgent"].noul)
print(response.choices["team"].choice, response.choices["team"].confidence)
print(response.scores["severity"].score)


a = {"state": {"query": "whats the best pizza place to eat in Oshawa", "passages":""},
 "questions": {"answerable": Noul(instructions="Can the query be answered using these Reddit comments?")}}

res = classifier.invoke({
  "state": "Fifa world cup final 2026 results",
  "questions": {
    "true_claim": Noul(instructions="Can this query be resolved by using reddit comments as a source?")
  }
})

