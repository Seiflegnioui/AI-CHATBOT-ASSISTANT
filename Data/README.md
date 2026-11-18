# Seif Eddine Legnioui — Personal RAG Knowledge Base

A source-grounded knowledge base for a portfolio chatbot that answers questions about Seif's background, education, professional experience, skills, and projects.

## Files

- `profile.md` — biography and career direction.
- `education.md` — academic history.
- `experience.md` — employment and freelance experience.
- `skills.md` — technologies and skills, with evidence levels.
- `projects.md` — projects and their technical details.
- `certifications.md` — certificates and learning milestones visible in public sources.
- `links.md` — public profile and repository links.
- `chatbot_instructions.md` — grounding and response rules for the chatbot.
- `structured_profile.json` — structured facts for exact questions (dates, status, counts).
- `sources_and_verification.md` — provenance, caveats, and facts to verify.

## Recommended ingestion

Load each Markdown file separately. Keep the filename and category as metadata, then split the loaded documents into chunks. Example metadata:

```json
{
  "source": "projects.md",
  "category": "projects",
  "owner": "Seif Eddine Legnioui"
}
```

Keep `structured_profile.json` outside the vector index or load it into a structured store. Use it for exact questions such as dates, counts, education status, and internship preferences. Semantic retrieval is not a reliable counting mechanism.

## Evidence policy

The November 2026 CV is the primary source for formal education, professional experience, and the two featured projects. Public LinkedIn and GitHub information supplements it. Prior project discussions are marked as ongoing or needing confirmation when they are not documented in the CV. A technology being mentioned does not prove expert-level proficiency.

Review `sources_and_verification.md` before making the chatbot public. In particular, confirm the latest internship status and which exploratory/academic projects should be presented as portfolio projects.
