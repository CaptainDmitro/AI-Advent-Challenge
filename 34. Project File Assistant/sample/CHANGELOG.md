# Changelog

## 0.2.0 (2026-05-14)

- Passwords are hashed with salted PBKDF2 (`hash_password`); `legacy_hash`
  (MD5) is deprecated.
- `PUT /notes/{note_id}` updates a note.

## 0.1.0 (2026-03-20)

- Sign-up, sign-in and the profile (`/users/register`, `/users/login`,
  `/users/me`).
- Notes: list, read, create.
