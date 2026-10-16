# Sync

## How sync works

Notes are saved on the device first and sent to Notely's servers in the
background, usually within a few seconds. Other devices signed in to the same
account get the change the next time they connect. The status icon at the
bottom of the sidebar shows "Synced", "Syncing…" or "Offline". Notes can be
edited offline; they sync when the connection is back.

## SYNC_DEVICE_LIMIT: too many devices on the Free plan

The Free plan syncs on **2 devices**. A third device can sign in and read the
notes that were already there, but its changes don't sync, and the status
shows "Device limit reached" with the code `SYNC_DEVICE_LIMIT`.

To fix it, either:

1. Remove a device you no longer use in **Settings → Devices → Remove**. The
   slot is free at once; the new device starts syncing within a minute.
2. Or upgrade to Pro, which syncs on unlimited devices.

Removing a device doesn't delete any notes.

## Conflicts

If the same note was changed on two devices while both were offline, Notely
keeps both versions: the later one stays as the note, and the earlier one is
saved next to it as "Note title (conflicted copy)". Merge them by hand and
delete the copy.

## Sync is stuck

Check that the device is online and signed in to the right account. Sign
out and back in to restart sync: unsynced notes are kept on the device and
sent after the sign-in. Notes larger than 25 MB (usually because of
attachments) don't sync; the note shows a warning.
