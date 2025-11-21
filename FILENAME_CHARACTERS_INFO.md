# Ongeldige karakters in bestandsnamen

## Algemeen ongeldige karakters (alle besturingssystemen):
- `<` (kleiner dan)
- `>` (groter dan)
- `:` (dubbele punt)
- `"` (aanhalingsteken)
- `/` (forward slash)
- `\` (backslash)
- `|` (pipe/verticale streep)
- `?` (vraagteken)
- `*` (asterisk)

## Windows-specifiek:
- Windows staat deze karakters NIET toe:
  - `< > : " / \ | ? *`
  - Ook niet toegestaan: controlekarakters (ASCII 0-31)
  - Bestandsnamen mogen niet eindigen met een punt (`.`) of spatie
  - Reserverde namen: `CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9`

## Linux/Mac-specifiek:
- `/` is niet toegestaan (directory separator)
- `\0` (null character) is niet toegestaan
- Meestal toleranter dan Windows

## In ons geval:
De huidige code gebruikt:
```javascript
const filename = `transcriptie-${timestamp}.txt`;
```

Waar `timestamp` is:
```javascript
new Date().toISOString().replace(/[:.]/g, "-")
```

Dit zou moeten werken omdat:
- `:` wordt vervangen door `-`
- `.` wordt vervangen door `-`
- ISO string bevat alleen: `2025-11-21T18-30-45-123Z` (na replace)

Maar we kunnen het veiliger maken door ALLE ongeldige karakters te verwijderen.

