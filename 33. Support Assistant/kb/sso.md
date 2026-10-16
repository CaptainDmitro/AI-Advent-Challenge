# Single sign-on (SSO)

Single sign-on lets the members of a Team workspace sign in to Notely with
their company account instead of a Notely password. It is available on the
**Team plan** only. Notely supports SAML 2.0 and OpenID Connect, and has
ready-made setups for Okta, Microsoft Entra ID and Google Workspace.

## Setting up SSO

A workspace admin opens **Admin console → Security → Single sign-on**,
verifies the company's email domain with a DNS TXT record, and connects the
identity provider. Until SSO is enforced, members can sign in either way:
with SSO or with their password.

## Enforced SSO

When the admin turns on **Enforce SSO**, every account whose email is on the
verified domain must sign in through the identity provider:

- Password sign-in is turned off for these accounts and fails with the code
  `SSO_REQUIRED`.
- "Forgot password?" doesn't send a reset email to these accounts.
- Sessions that were already open stay signed in until they expire.

Members get an email when SSO becomes required, but it is easy to miss, so
the usual first sign of enforcement is "my password stopped working" right
after the change.

## Signing in with SSO

Click **Sign in with SSO** on the sign-in page (in the apps it is under the
password field), enter the work email, and sign in at the identity provider.
The notes, notebooks and shared spaces are the same as before: only the way
of signing in changes.

## When SSO sign-in fails

- "You are not assigned to this application": the company's IT admin has to
  give you access to Notely in the identity provider (for example, add you to
  the Notely app in Okta).
- "Domain not verified" or a loop back to the sign-in page: the workspace
  admin has to check the SSO setup in the admin console.
- Notely support can't turn SSO off for one member: it is the workspace
  admin's setting. Support can tell you who the admins of your workspace are.
