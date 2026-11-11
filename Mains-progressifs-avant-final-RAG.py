# --- 4. etape 2 test du chunk : lecture des 3 docs
#def main() -> None:
#    chunks = charger_documents(DOSSIER_DOCS)
#    print(f"{len(chunks)} chunks chargés :")
#    for c in chunks:
#        print(f"  - [{c['source']}] {c['texte'][:60]}...")

# --- 5. etape 3 test final avec embeddings par mots
def main() -> None:
    chunks = charger_documents(DOSSIER_DOCS)
    print(f"{len(chunks)} chunks chargés, calcul des embeddings...")
    index = calculer_embeddings([c["texte"] for c in chunks])
    print("Index prêt. Posez une question de recherche (quitter pour sortir).\n")
    while True:
        question = input("Recherche > ").strip()
        if question.lower() in ("quitter", "exit", "q"):
            break
        if not question:
            continue
        for chunk, score in rechercher(question, chunks, index):
            print(f"\n  [{score:.3f}] ({chunk['source']}) {chunk['texte']}")
        print()

# etape 1 : requete hallucination
# def main() -> None:
#    print("=== Assistant Appels d'Offres (étape 1 : branchement API) ===")
#    print("Tapez 'quitter' pour sortir.\n")
#    while True:
#        question = input("Votre question > ").strip()
#        if question.lower() in ("quitter", "exit", "q"):
#            break
#        if not question:
#            continue
#        print("\n" + poser_question(question) + "\n")
