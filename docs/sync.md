# How sync works

Lumen syncs a notebook through a relay called Harbor. The app encrypts each note on the device before upload. Harbor stores ciphertext and does not have the notebook key.

# Conflicts

If two devices edit the same note while offline, Lumen keeps both versions and asks you to pick one. It does not merge sentences automatically.
