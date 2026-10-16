# Two-factor authentication

Two-factor authentication (2FA) adds a six-digit code to every sign-in with a
password. Notely supports authenticator apps that use TOTP codes: Google
Authenticator, Microsoft Authenticator, 1Password, Authy and others. It is
available on every plan, and a Team admin can require it for all members.

## Turning on two-factor authentication

1. Open **Settings → Security → Two-factor authentication** and click
   **Turn on**.
2. Scan the QR code with the authenticator app.
3. Enter the six-digit code the app shows to confirm.
4. Save the ten **recovery codes**. Each one works once, instead of a code
   from the app.

From then on Notely asks for a code after the password on every new device.
A device you trust can be remembered for 30 days.

## Codes are rejected: check the device clock

A TOTP code is calculated from a secret and the **current time on the
phone**. It changes every 30 seconds. Notely accepts the current code and the
ones just before and after it, so the phone's clock may be about 30 seconds
off. If the clock is off by a minute or more, **every code is rejected** with
`INVALID_2FA_CODE`, even though the app shows a code and the secret is
right. This happens after travel across time zones, after a battery ran
flat, or when the date and time are set by hand.

How to fix it:

- **iPhone**: Settings → General → Date & Time → turn on **Set
  Automatically**.
- **Android**: Settings → System → Date & time → turn on **Set time
  automatically**. In Google Authenticator also use the menu → **Time
  correction for codes** → **Sync now**.
- Then wait for a new code to appear and enter it right away.

Support agents can see this case in the sign-in history: the failed attempts
have the code `INVALID_2FA_CODE` and a detail about clock skew. Turning 2FA
off is not needed and is not the fix.

## Recovery codes

If the phone is not at hand, click **Use a recovery code** on the code screen
and enter one of the ten recovery codes saved when 2FA was turned on. Each
code works once. New codes can be generated in **Settings → Security**, which
makes the old ones invalid.

## Lost phone

Sign in with a recovery code, then turn 2FA off and on again in **Settings →
Security** to connect the new phone. Without recovery codes, support can
remove 2FA only after confirming that you own the account: the request has to
come from the account's email address, and it takes up to 2 business days.
