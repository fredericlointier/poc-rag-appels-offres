# =============================================================
# POC — Assistant Réponses à appels d'offres (RAG basique)
# Frederic Lointier / Stack : Python + API Mistral
#
# ETAPE 1 : brancher l'API et dialoguer avec le modèle stdd Mistral
# ETAPE 2 : charger 3 documents et les découper (chunking)
# ETAPE 3 : recherche par embeddings (note statistique semantique)
# ETAPE 4 : RAG complet avec citation des sources
#
# INSTALLATION (une seule fois) :
#   pip3 install mistralai
#   + upgrade 26.3 : "pip3 install --upgrade pi"
#
# CLE API : dans une variable d'environnement.
#   Terminal en 2 prompts :
#     echo 'export MISTRAL_API_KEY="..."' >> ~/.zshrc
#     source ~/.zshrc
# =============================================================

import os
from mistralai.client import Mistral

# --- 1. Connexion à l'API -------------------------------------
# On lit la clé dans l'environnement (jamais en dur dans le code :
# le fichier sera poussé sur GitHub, la clé doit rester secrète).
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# --- 2. Le prompt système -------------------------------------
# C'est le "cahier des charges permanent" du modèle.
# on a ajouté la règle : "réponds UNIQUEMENT
# à partir des extraits fournis et cite tes sources".
PROMPT_SYSTEME = (
    "Tu es un assistant d'aide à la réponse aux appels d'offres. "
    "Tu réponds UNIQUEMENT à partir des extraits de documents fournis. "
    "Cite systématiquement la source entre crochets, par exemple "
    "[conditions_administratives.txt], après chaque information. "
    "Si la réponse ne se trouve pas dans les extraits, réponds "
    "exactement : 'Information absente des documents de référence', "
    "sans rien inventer ni compléter de mémoire."
)

# --- 3. La boucle de dialogue ---------------------------------
def poser_question(question: str) -> str:
    """Envoie la question au modèle et renvoie sa réponse."""
    reponse = client.chat.complete(
        model="mistral-medium-latest",
        messages=[
            {"role": "system", "content": PROMPT_SYSTEME},
            {"role": "user", "content": question},
        ],
    )
    return reponse.choices[0].message.content

# --- 4. Chargement des documents et chunking ------------------
DOSSIER_DOCS = "docs_ao"   # dossier à côté du script

def charger_documents(dossier: str) -> list[dict]:
    """Charge chaque fichier texte et le découpe en chunks
    (un paragraphe = un chunk). Chaque chunk garde sa source."""
    chunks = []
    for nom_fichier in sorted(os.listdir(dossier)):
        if not nom_fichier.endswith((".txt", ".md")):
            continue
        chemin = os.path.join(dossier, nom_fichier)
        with open(chemin, encoding="utf-8") as f:
            texte = f.read()
        for paragraphe in texte.split("\n\n"):
            if paragraphe.strip():
                chunks.append({
                    "source": nom_fichier,
                    "texte": paragraphe.strip(),
                })
    return chunks

# --- 5. Recherche par embeddings ---------------------------------
MODELE_EMBEDDING = "mistral-embed"

def calculer_embeddings(textes: list[str]) -> list[list[float]]:
    """Demande à Mistral la 'carte du sens' de chaque texte.
    Un seul appel API pour tous les textes (économie + débit)."""
    reponse = client.embeddings.create(
        model=MODELE_EMBEDDING,
        inputs=textes,
    )
    return [d.embedding for d in reponse.data]


def similarite(a: list[float], b: list[float]) -> float:
    """Proximité du sens entre deux cartes (1 = identique, 0 = rien à voir)."""
    num = sum(x * y for x, y in zip(a, b))
    norme_a = sum(x * x for x in a) ** 0.5
    norme_b = sum(x * x for x in b) ** 0.5
    return num / (norme_a * norme_b)


def rechercher(question: str, chunks: list[dict], index: list[list[float]], k: int = 2):
    """Renvoie les k chunks les plus proches du sens de la question."""
    vecteur_question = calculer_embeddings([question])[0]
    scores = [similarite(vecteur_question, v) for v in index]
    classement = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [(chunks[i], scores[i]) for i in classement[:k]]

# --- 6. La réponse agrégeant toutes les étapes précédentes -------
def repondre(question: str, chunks: list[dict], index: list[list[float]], k: int = 3):
    """RAG complet : recherche les passages pertinents, les injecte
    dans la question, et fait rédiger la réponse sourcée par le modèle."""
    resultats = rechercher(question, chunks, index, k=k)
    extraits = "\n\n".join(
        f"[{chunk['source']}]\n{chunk['texte']}" for chunk, _ in resultats
    )
    reponse = client.chat.complete(
        model="mistral-medium-latest",
        messages=[
            {"role": "system", "content": PROMPT_SYSTEME},
            {"role": "user", "content": f"Extraits de documents :\n\n{extraits}\n\nQuestion : {question}"},
        ],
    )
    return reponse.choices[0].message.content, resultats


# --- 7. Le main agrégeant toutes les étapes précédentes -------
def main() -> None:
    chunks = charger_documents(DOSSIER_DOCS)
    print(f"{len(chunks)} chunks chargés, calcul des embeddings...")
    index = calculer_embeddings([c["texte"] for c in chunks])
    print("Assistant prêt. Posez vos questions (quitter pour sortir).\n")
    while True:
        question = input("Question > ").strip()
        if question.lower() in ("quitter", "exit", "q"):
            break
        if not question:
            continue
        reponse, resultats = repondre(question, chunks, index)
        print("\n" + reponse)
        print("\n  Sources consultées :")
        for chunk, score in resultats:
            print(f"   - [{chunk['source']}] (score {score:.3f})")
        print()

if __name__ == "__main__":
    main()
