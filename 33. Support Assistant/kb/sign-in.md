# Signing in to Notely

Notely accounts work the same on the web app (app.notely.app), the desktop
apps (macOS, Windows, Linux) and the mobile apps (iOS, Android). There are
three ways to sign in:

- **Email and password**, available on every plan.
- **Continue with Google**, for accounts created with a Google address.
- **Sign in with SSO**, for Team workspaces whose admin connected an identity
  provider (Okta, Microsoft Entra ID, Google Workspace). See "Single sign-on".

When a sign-in fails, Notely shows a short message and records an error code
in the account's sign-in history. Support agents see the same code next to
every failed attempt, so the code is the fastest way to find the cause.

## Error codes at sign-in

| Code | What it means | Section |
|---|---|---|
| `WRONG_PASSWORD` | the email or password is incorrect | WRONG_PASSWORD |
| `ACCOUNT_LOCKED` | too many failed attempts, the account is locked | ACCOUNT_LOCKED |
| `EMAIL_NOT_VERIFIED` | the email address was never confirmed | EMAIL_NOT_VERIFIED |
| `SSO_REQUIRED` | the organization turned off password sign-in | SSO_REQUIRED |
| `INVALID_2FA_CODE` | the two-factor code was rejected | INVALID_2FA_CODE |
| `ACCOUNT_SUSPENDED` | the account was suspended (billing or abuse) | ACCOUNT_SUSPENDED |

## WRONG_PASSWORD: the email or password is incorrect

The password doesn't match the account, or there is no account with that
email. Passwords are case-sensitive, so check Caps Lock and the keyboard
layout. If you signed up with "Continue with Google", the account has no
password: use the Google button instead.

To set a new password, click **Forgot password?** on the sign-in page and
enter your email. The reset link arrives from `no-reply@notely.app` and is
valid for 1 hour. Every failed attempt counts towards the lock described in
ACCOUNT_LOCKED, so after two or three failures it is better to reset the
password than to keep guessing.

## ACCOUNT_LOCKED: too many failed sign-in attempts

Notely locks an account to protect it from password guessing:

- After **5 failed attempts in a row** the account is locked for
  **30 minutes**. It unlocks by itself after that.
- After **10 failed attempts within 24 hours** the account stays locked
  **until the password is reset**. Waiting does not help.

While the account is locked, every sign-in shows "Your account is temporarily
locked", even with the correct password, on every device.

How to unlock it:

1. Click **Forgot password?** on the sign-in page and enter the account's
   email.
2. Open the reset link (valid for 1 hour) and choose a new password.
3. The reset unlocks the account at once and resets the failed-attempt
   counter. Sign in again on each device with the new password.

Support agents can't lift the lock by hand: resetting the password is the
fastest way. If you didn't make these attempts yourself, someone may be
guessing your password: after the reset, turn on two-factor authentication.

## EMAIL_NOT_VERIFIED: the email address is not confirmed

Every new account has to confirm its email before the first sign-in. Right
after sign-up Notely sends a confirmation email from `no-reply@notely.app`
with the subject "Confirm your Notely account". The link in it is valid for
**24 hours**. Until the email is confirmed, sign-in fails with
"Please confirm your email address".

If the email didn't arrive or the link expired:

1. On the sign-in page, enter your email and click **Resend confirmation
   email**. A new link replaces the old one.
2. Check the Spam, Promotions and Updates folders. Corporate mail filters
   sometimes hold the message for a few minutes.
3. Make sure the address has no typo. If it does, sign up again with the
   right address: an unconfirmed account is deleted after 7 days.

## SSO_REQUIRED: your organization requires single sign-on

When the admin of a Team workspace turns on **Enforce SSO** for the company's
email domain, password sign-in is switched off for every account on that
domain. From then on, the email and password are rejected with "Your
organization requires single sign-on", even if the password is correct.
Resetting the password does not help, and the reset email is not sent.

To sign in:

1. On the sign-in page, click **Sign in with SSO**.
2. Enter your work email. Notely sends you to your company's identity
   provider (for example Okta).
3. Sign in there with your company account. You are returned to Notely,
   with all your notes.

On mobile and desktop apps the same **Sign in with SSO** button is under the
password field. If the identity provider says that you are not assigned to
Notely, only your company's IT admin can fix it. See "Single sign-on".

## INVALID_2FA_CODE: the two-factor code is rejected

The password was right but the six-digit code from the authenticator app was
not accepted. The most common cause is the phone's clock: codes are
calculated from the current time, and a clock that is off by more than a
minute makes every code wrong. See "Two-factor authentication" → "Codes are
rejected: check the device clock". If the phone is lost, use a recovery code.

## ACCOUNT_SUSPENDED: the account was suspended

An account is suspended when a payment failed for more than 14 days on a paid
plan, or after a violation of the terms of use. The notes are kept. A
payment problem is fixed by updating the card in the billing portal (the link
is in the email about the failed payment). For any other suspension, contact
support.

## Signed out unexpectedly

A session lasts 30 days on the web and 90 days in the apps. Changing or
resetting the password signs out every other device. An admin of a Team
workspace can also sign out all members. Sign in again; your notes are kept
on the server.
