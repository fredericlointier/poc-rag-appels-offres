Assistant de réponse aux appels d'offres (POC RAG)

Assistant en ligne de commande qui répond aux questions des rédacteurs de réponses à appels d'offres en s'appuyant uniquement sur les documents de référence, en citant systématiquement ses sources, et en refusant de répondre lorsque l'information est absente.

POC réalisé avec l'API Mistral (modèles mistral-medium-latest et mistral-embed), à visée d'apprentissage et de démonstration.

A. Le problème résolu
Un LLM seul, interrogé sur un métier, produit une réponse plausible... mais inventée. Exemple réel, avant RAG, à la question « Quelle garantie par le candidat ? » :
- Garantie de soumission, garantie de bonne fin, garantie de performance, garantie financière, garantie légale de conformité...
-> Réponse de manuel, hors sujet. 
Après RAG, avec les mêmes documents :
- Le titulaire du marché garantit ses livraisons pendant 12 mois à compter de la fin de la recette, au prononcé de la VSR [conditions_administratives.txt].

B. Architecture en 4 briques
	1	Chunking : les documents de référence (docs_ao/*.txt) sont découpés paragraphe par paragraphe, chaque chunk conservant son fichier source.
	2	Embeddings : chaque chunk est vectorisé avec mistral-embed (l'index est calculé une fois au démarrage).
	3	Recherche sémantique : la question est vectorisée à son tour ; les 3 chunks les plus proches du sens sont sélectionnés par similarité cosinus — même sans aucun mot en commun avec la question.
	4	RAG sourcé : les passages trouvés sont injectés dans le prompt, cadré avec une consigne système stricte : « répondre uniquement à partir des extraits, citer la source, et répondre « Information absente des documents de référence » si l'information ne s'y trouve pas ».

C. Installation
# 1. Cloner puis créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 2. Installer les dépendances
pip install mistralai

# 3. Définir sa clé API (console.mistral.ai) — jamais dans le code
export MISTRAL_API_KEY="votre_cle"

# 4. Lancer
python assistant_ao.py

Compatible Python 3.10+. Tester avec le package mistralai 3.7 (la classe Mistral y est importée depuis mistralai.client ; avec les versions 1.x, l'import se fait directement depuis mistralai).

D. Utilisation
Déposer ses documents de référence au format texte dans docs_ao/ (paragraphes séparés par des lignes vides), puis poser ses questions dans le terminal :
Question > quelle garantie par le candidat ?
Question > quel est le montant de la caution bancaire exigée ?
Réponse attendue pour la seconde : « Information absente des documents de référence » — c'est un critère d'acceptation du POC, pas un défaut.

E. Enseignements du développement
	•	La qualité du RAG tient au duo recherche + rédaction : le classement des chunks peut être serré (scores proches sur un corpus court) ; injecter les 3 meilleurs candidats et laisser le modèle choisir dans les extraits est plus robuste que de n'injecter que le premier.
	•	Le prompt système est l'outil de conformité : exiger la citation des sources et le refus explicite en cas d'absence d'information élimine l'essentiel des hallucinations sur ce cas d'usage.
	•	Bonnes pratiques appliquées : clé API en variable d'environnement, environnement virtuel isolé, un seul appel d'embeddings pour tout le corpus, gestion des erreurs d'API (429 : limite de débit, réessayer après pause).
Pistes d'amélioration
	•	Ajouter un reranker pour fiabiliser le classement des passages sur un corpus plus grand.
	•	Découpage par sections titrées plutôt que par paragraphes.
	•	Réessai automatique avec temporisation sur les erreurs 429.
	•	Interface web minimaliste (streamlit) pour une démonstration client.
