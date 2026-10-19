# Project rules

Every change to Notely API must keep these true. Reviews check them, and so
does the assistant when asked.

1. **Auth on notes.** Every endpoint in `app/routes/notes.py` declares
   `Depends(require_auth)`.
2. **Documented API.** Every endpoint in `app/routes/` has a section in
   `docs/api.md` with the same method and path.
3. **No secrets in code.** Keys and passwords come from the environment
   (`os.environ`), never from a literal in `app/`.
4. **One way to hash passwords.** Only `hash_password` and
   `verify_password`; `legacy_hash` is deprecated and must not be called.
5. **Tests per router.** Every module in `app/routes/` has a
   `tests/test_<module>.py`.
6. **Logging, not print.** Code in `app/` uses `logging`, never `print(`.
7. **Changelog.** A user-visible change gets a line in `CHANGELOG.md` under
   `## Unreleased`.
