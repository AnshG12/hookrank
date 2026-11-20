import pytest

from hookrank.parsing import parse_episode

SCRIPT = """EPISODE NUMBER: Episode {n}
EPISODE LOGLINE: Juliana discovers that Clarissa stole her groom.
SCENE 01
Scene heading: INT. CHAPEL - DAY
Scene start: 00:00.00
Scene end: 01:30.00

**CAMERA:**
Start: (00:00.00) | End: (00:03.00)
Wide shot of the chapel doors opening.

**LADY IN GOLD GOWN (Smug):**
Start: (00:03.00) | End: (00:06.00)
"Because he is marrying me, Clarissa, not you."

**JULIANA (Shocked):**
Start: (00:06.50) | End: (00:09.00)
"Gerald, how could you betray me?"

**ACTION:**
Start: (00:09.00) | End: (00:14.00)
Juliana slaps Gerald across the face.

**GERALD (Cold):**
Start: (00:14.00) | End: (00:20.00)
"I never loved you, Juliana."

**CAMERA:**
Start: (00:20.00) | End: (00:30.00)
Slow push-in on Juliana's trembling face.

**JULIANA (Quiet):**
Start: (00:30.00) | End: (00:36.00)
"Then who sent the letter that brought me here?"

**CAMERA:**
Start: (00:36.00) | End: (00:38.00)
Close-up on the sealed letter. A cliffhanger chord swells.

**GERALD (Relieved):**
Start: (00:38.50) | End: (00:42.00)
"It was your father. Everything is settled now."
"""


@pytest.fixture
def episode():
    return parse_episode(SCRIPT.format(n=1), "episode-001.txt", "test-series")


@pytest.fixture
def episode_factory():
    def make(n: int):
        return parse_episode(SCRIPT.format(n=n), f"episode-{n:03d}.txt", "test-series")
    return make
