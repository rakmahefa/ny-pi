# Gemini Web Provider — Guide

Ce guide explique l'utilisation du provider `gemini-web` ajouté sur la branche `feat/gemini-cookie-provider`.

Le provider reproduit le transport Gemini Web utilisé par `rakmahefa/ny-gemini-acp`. Il ne passe pas par l'API Gemini publique : il utilise la session Google portée par `cookie.json`, récupère les tokens de page `/app`, puis appelle le endpoint Web `StreamGenerate`.

## 1. Pré-requis

- Node.js 22+
- Le projet `ny-pi` installé et ses dépendances npm disponibles
- Une session Gemini Web Google déjà authentifiée
- Un export des cookies Google utilisé par Gemini Web

## 2. Installer `cookie.json`

Place le fichier dans l'un de ces emplacements :

```text
./cookie.json
./vendor/cookie.json
```

Ou indique explicitement le chemin avec :

```bash
export GEMINI_COOKIE_FILE="/chemin/vers/cookie.json"
```

Le fichier n'a pas besoin d'être ajouté au dépôt. Les chemins `cookie.json`, `vendor/cookie.json` et `**/cookie.json` sont ignorés par Git.

## 3. Formats acceptés

Le provider accepte les mêmes familles de formats que `ny-gemini-acp`.

### Export EditThisCookie

```json
[
  {
    "name": "SID",
    "value": "...",
    "domain": ".google.com",
    "expirationDate": 9999999999
  },
  {
    "name": "SAPISID",
    "value": "...",
    "domain": ".google.com",
    "expirationDate": 9999999999
  }
]
```

### Format objet

```json
{
  "cookie": "SID=...; SAPISID=...; __Secure-1PSID=...",
  "sapisid": "..."
}
```

### Chaîne brute

```text
SID=...; SAPISID=...; __Secure-1PSID=...
```

Le parser ignore les cookies qui ne ciblent pas `google.com` ou ses sous-domaines et écarte les cookies expirés.

## 4. Authentification dans `ny-pi`

Le provider apparaît sous l'identifiant :

```text
provider: gemini-web
```

L'authentification est une auth de type `api_key` au niveau de l'infrastructure de `ny-pi`, mais la valeur stockée est uniquement **le chemin du fichier cookie**, jamais le contenu des cookies.

Connexion interactive :

```text
/login
```

Puis choisir `Gemini Web` et fournir le chemin vers `cookie.json`.

Pour une configuration ambient, `GEMINI_COOKIE_FILE` est prioritaire. En absence de variable d'environnement, le provider recherche automatiquement `./cookie.json` puis `./vendor/cookie.json`.

## 5. Modèles disponibles

Le catalogue porté depuis `ny-gemini-acp` comprend notamment :

```text
 gemini-3.6-flash
 gemini-3.5-flash
 gemini-3.5-flash-thinking
 gemini-3.1-pro
 gemini-3.1-pro-enhanced
 gemini-auto
 gemini-3.5-flash-thinking-lite
 gemini-flash-lite
```

Le provider utilise `gemini-3.6-flash` comme fallback du mapping Web lorsque le modèle demandé n'est pas reconnu.

## 6. Thinking

Le provider accepte un niveau Web `0..4`.

Une forme compatible avec le mapping de `ny-gemini-acp` est :

```text
model@think=2
```

Exemple :

```text
 gemini-3.5-flash@think=2
```

Le niveau est borné à `4`.

## 7. Transport Web

Le provider effectue, dans cet ordre logique :

1. lecture de `cookie.json` ;
2. construction de l'en-tête `Cookie` ;
3. récupération du cookie `SAPISID` ;
4. calcul de `SAPISIDHASH` pour `https://gemini.google.com` ;
5. `GET https://gemini.google.com[/u/<user>]/app` pour récupérer les tokens de page ;
6. construction du payload `f.req` ;
7. `POST` vers `_/BardChatUi/data/assistant.lamda.BardFrontendService/StreamGenerate` ;
8. décodage défensif des frames `wrb.fr` ;
9. projection vers les événements `ny-pi` (`text_*`, `toolcall_*`, `done`, `error`).

Les tokens de page sont mémorisés temporairement et renouvelés après leur TTL.

## 8. Sécurité

Ne jamais committer `cookie.json`.

Vérifie avant un commit :

```bash
git status --short
git check-ignore -v cookie.json
```

Le résultat attendu de `git check-ignore` doit pointer vers `.gitignore`.

Ne copie jamais les valeurs réelles de `SID`, `SAPISID`, `__Secure-*`, etc. dans une issue, un log, un commit ou une PR.

Si la session Google doit être renouvelée, remplace uniquement le fichier local `cookie.json`.

## 9. Tests

Tests unitaires :

```bash
npm test -- packages/ai/test/gemini-web.test.ts
```

Vérification TypeScript / build :

```bash
npm run check
npm run build
```

La CI GitHub Actions du dépôt exécute également le build, le check et les tests sur les pull requests.

## 10. Debug local

Pour vérifier uniquement la détection du fichier cookie sans envoyer son contenu au dépôt :

```bash
export GEMINI_COOKIE_FILE="$PWD/cookie.json"
```

Puis vérifier que le provider est disponible dans l'environnement `ny-pi`.

En cas d'erreur d'authentification :

```text
No usable Google cookies found
```

le premier point à vérifier est l'expiration ou le domaine des cookies.

En cas de `401`, `403` ou refus du backend Gemini Web, régénérer un export de session Google peut être nécessaire.

## 11. Référence d'architecture

Le portage suit le contrat du crate `llm-provider` de `rakmahefa/ny-gemini-acp` :

```text
cookie.json
   │
   ▼
Cookie parser
   │
   ├── Cookie header
   └── SAPISIDHASH
   │
   ▼
Gemini Web /app
   │
   └── page tokens
   │
   ▼
StreamGenerate
   │
   ▼
wrb.fr frame decoder
   │
   ├── text
   ├── tool call
   └── metadata
   │
   ▼
ny-pi AssistantMessageEventStream
```

## 12. Important

Ce provider dépend d'une interface Web Gemini non publique et susceptible d'évoluer indépendamment de l'API Gemini officielle. Une modification du backend Gemini peut donc nécessiter une mise à jour du payload, du mapping des modèles ou du décodage des frames.
