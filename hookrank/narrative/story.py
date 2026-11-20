"""Series-level story model, assembled deterministically from per-episode extractions.

Characters. Speaker labels vary across a series (JULIANA / LADY JULIANA; LADY IN GOLD GOWN
is CLARISSA; MYSTERIOUS BEARDED MAN is later named). Identity is resolved with a union-find
over label and name nodes, using three kinds of edge:
  1. verified label->name claims from extraction (the name must appear in that episode's text),
  2. identical text (label "CLARISSA" and name "Clarissa"),
  3. honorific stripping, only when the stripped form already exists as its own node
     ("LADY JULIANA" -> "JULIANA" because JULIANA exists; "LADY ASHCROFT" and
     "LORD ASHCROFT" stay apart because "ASHCROFT" never occurs alone).
Unions are conflict-aware: two components that each carry different personal names are
never merged; the refused merge is logged. One bad claim cannot fuse two characters.

Threads. Episode threads are linked into series threads by participant overlap and kind
(rules in ``_link_score``). This is a heuristic; every link is recorded with its score. A
single shared participant is never enough for multi-character threads: the protagonist is
in almost every thread, so sharing her says nothing (observed: Juliana's personal resolve
arc wrongly joined to the poisoning plot before this rule).

Event characters are computed here too: the speaker plus verified personal names mentioned
in the event text. Deterministic and checkable, unlike a model's list of "characters involved".
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from ..parsing import chronology_key
from ..schemas import Episode
from .schemas import ConsensusEvent, EpisodeStory, Issue
from .text import name_in_text, normalize

HONORIFICS = {
    "the", "lady", "lord", "sir", "dame", "mr", "mrs", "ms", "miss", "master", "mistress", "madam", "madame",
    "dr", "doctor", "duke", "duchess", "king", "queen", "prince", "princess", "count", "countess", "baron",
    "baroness", "earl", "father", "captain", "general", "young", "old", "dowager",
}
PRINCIPAL_MIN_LINES = 10
PRINCIPAL_MIN_EPISODES = 3


class Character(BaseModel):
    character_id: str
    name: str
    labels: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    descriptions: dict[str, str] = Field(default_factory=dict)   # episode_number -> description
    episodes: list[str] = Field(default_factory=list)            # episode numbers where they speak
    lines: int = 0
    first_episode: str = ""
    principal: bool = False


class Relationship(BaseModel):
    a: str            # character_id (or raw name if unresolved)
    b: str
    relation: str
    change: str = ""
    episode_number: str
    evidence: list[str] = Field(default_factory=list)


class ThreadEntry(BaseModel):
    episode_number: str
    episode_id: str
    local_key: str
    status: str
    summary: str
    link_score: float | None = None   # None for the entry that created the thread
    evidence: list[str] = Field(default_factory=list)


class SeriesThread(BaseModel):
    thread_id: str
    kind: str
    participants: list[str] = Field(default_factory=list)    # character_ids
    entries: list[ThreadEntry] = Field(default_factory=list)

    @property
    def last_episode(self) -> str:
        return self.entries[-1].episode_number if self.entries else ""


class SeriesStory(BaseModel):
    series_id: str
    characters: list[Character] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    threads: list[SeriesThread] = Field(default_factory=list)
    thread_map: dict[str, str] = Field(default_factory=dict)      # "episode_id:T1" -> thread_id
    episode_summaries: dict[str, str] = Field(default_factory=dict)
    issues: list[Issue] = Field(default_factory=list)


# ---- union-find with conflict-aware merging ------------------------------------------------

def _core(text: str) -> str:
    """Name with leading honorifics removed ('lady juliana' -> 'juliana')."""
    words = normalize(text).split()
    while len(words) > 1 and words[0] in HONORIFICS:
        words = words[1:]
    return " ".join(words)


@dataclass
class _Resolver:
    parent: dict[str, str] = field(default_factory=dict)
    names: dict[str, set[str]] = field(default_factory=dict)    # root -> core personal names
    refused: list[str] = field(default_factory=list)

    def add(self, node: str, personal_name: str | None = None) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.names[node] = set()
        if personal_name:
            self.names[self.find(node)].add(_core(personal_name))

    def find(self, node: str) -> str:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, a: str, b: str, reason: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        na, nb = self.names[ra], self.names[rb]
        if na and nb and not (na & nb):
            self.refused.append(f"{a} + {b} ({reason}): would merge {sorted(na)} with {sorted(nb)}")
            return False
        self.parent[rb] = ra
        self.names[ra] |= self.names.pop(rb)
        return True


def _label_node(label: str) -> str:
    return "label:" + normalize(label)


def _name_node(name: str) -> str:
    return "name:" + normalize(name)


class CharacterIndex:
    """Maps any label or name used in the series to a canonical character."""

    def __init__(self, resolver: _Resolver, root_to_id: dict[str, str], id_to_name: dict[str, str]):
        self._resolver, self._root_to_id, self._id_to_name = resolver, root_to_id, id_to_name
        # Surface forms for mention detection: verified personal names and their honorific-free
        # cores only (generic labels such as "MAID" or "LORD" would match ordinary words).
        self._forms: dict[str, str] = {}
        for node in resolver.parent:
            if node.startswith("name:"):
                cid = root_to_id.get(resolver.find(node))
                text = node.split(":", 1)[1]
                for form in {text, _core(text)}:
                    if cid and form not in HONORIFICS:
                        self._forms.setdefault(form, cid)

    def mentions(self, text: str) -> set[str]:
        return {cid for form, cid in self._forms.items() if name_in_text(form, text)}

    def resolve(self, text: str) -> str | None:
        for node in (_label_node(text), _name_node(text), _label_node(_core(text)), _name_node(_core(text))):
            if node in self._resolver.parent:
                return self._root_to_id.get(self._resolver.find(node))
        return None

    def name(self, character_id: str) -> str:
        return self._id_to_name.get(character_id, character_id)


def _display(text: str) -> str:
    return text.title() if text.isupper() else text


def build_series_story(series_id: str, episodes: list[Episode], stories: dict[str, EpisodeStory],
                       consensus: dict[str, list[ConsensusEvent]]) -> tuple[SeriesStory, CharacterIndex]:
    ordered = sorted(episodes, key=chronology_key)
    resolver = _Resolver()
    line_counts: Counter[str] = Counter()
    speaking_eps: dict[str, list[str]] = defaultdict(list)

    for ep in ordered:
        for event in ep.events:
            if event.speaker:
                node = _label_node(event.speaker)
                resolver.add(node)
                line_counts[node] += 1
                if ep.episode_number not in speaking_eps[node]:
                    speaking_eps[node].append(ep.episode_number)
    claims: list[tuple[str, str, str]] = []
    for ep in ordered:
        story = stories.get(ep.episode_id)
        if not story or not story.ok:
            continue
        for c in story.characters:
            resolver.add(_label_node(c.label))
            if c.name_verified:
                resolver.add(_name_node(c.name), personal_name=c.name)
                claims.append((_label_node(c.label), _name_node(c.name), f"{ep.episode_number} extraction"))
    for a, b, reason in claims:                         # 1. verified label -> name claims
        resolver.union(a, b, reason)
    for node in list(resolver.parent):                  # 2. identical text across label/name
        kind, text = node.split(":", 1)
        twin = ("name:" if kind == "label" else "label:") + text
        if twin in resolver.parent:
            resolver.union(node, twin, "identical text")
    for node in list(resolver.parent):                  # 3. honorific stripping onto existing nodes
        kind, text = node.split(":", 1)
        core = _core(text)
        if core != text:
            for target in ("label:" + core, "name:" + core):
                if target in resolver.parent:
                    resolver.union(node, target, f"honorific '{text}' -> '{core}'")

    # Build canonical characters.
    members: dict[str, list[str]] = defaultdict(list)
    for node in resolver.parent:
        members[resolver.find(node)].append(node)
    episode_rank = {ep.episode_number: i for i, ep in enumerate(ordered)}
    characters: list[Character] = []
    root_to_id: dict[str, str] = {}
    descriptions: dict[str, dict[str, str]] = defaultdict(dict)
    name_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for ep in ordered:
        story = stories.get(ep.episode_id)
        if not story or not story.ok:
            continue
        for c in story.characters:
            root = resolver.find(_label_node(c.label))
            if c.description:
                descriptions[root].setdefault(ep.episode_number, c.description)
            if c.name_verified:
                name_votes[root][c.name] += 1
    roots = sorted(members, key=lambda r: (-sum(line_counts[n] for n in members[r]), r))
    for index, root in enumerate(roots, 1):
        labels = sorted({n.split(":", 1)[1].upper() for n in members[root] if n.startswith("label:")})
        names = sorted({n.split(":", 1)[1] for n in members[root] if n.startswith("name:")})
        lines = sum(line_counts[n] for n in members[root])
        eps = sorted({e for n in members[root] for e in speaking_eps.get(n, [])}, key=lambda e: episode_rank.get(e, 10**9))
        if name_votes[root]:
            display = name_votes[root].most_common(1)[0][0]
        else:
            top_label = max((n for n in members[root] if n.startswith("label:")),
                            key=lambda n: line_counts[n], default=members[root][0])
            display = top_label.split(":", 1)[1].upper()
        cid = f"C{index:02d}"
        root_to_id[root] = cid
        characters.append(Character(
            character_id=cid, name=_display(display), labels=labels, names=[_display(n) for n in names],
            descriptions=descriptions[root], episodes=eps, lines=lines, first_episode=eps[0] if eps else "",
            principal=lines >= PRINCIPAL_MIN_LINES or len(eps) >= PRINCIPAL_MIN_EPISODES))
    index = CharacterIndex(resolver, root_to_id, {c.character_id: c.name for c in characters})
    series = SeriesStory(series_id=series_id, characters=characters)
    series.issues += [Issue(stage="series", model="python", kind="merge_refused", detail=r) for r in resolver.refused]

    # Relationships and threads, chronologically.
    for ep in ordered:
        story = stories.get(ep.episode_id)
        if not story or not story.ok:
            continue
        series.episode_summaries[ep.episode_number] = story.summary
        for r in story.relationships:
            series.relationships.append(Relationship(
                a=index.resolve(r.a) or r.a, b=index.resolve(r.b) or r.b, relation=r.relation, change=r.change,
                episode_number=ep.episode_number, evidence=r.evidence))
        for t in story.threads:
            participants = sorted({cid for p in t.participants if (cid := index.resolve(p))})
            unresolved = [p for p in t.participants if not index.resolve(p)]
            if unresolved:
                series.issues.append(Issue(stage="series", model=story.model, kind="unresolved_participant",
                                           detail=f"{ep.episode_number} {t.key}: {unresolved}"))
            entry = ThreadEntry(episode_number=ep.episode_number, episode_id=ep.episode_id, local_key=t.key,
                                status=t.status, summary=t.summary, evidence=t.evidence)
            # Strongest link wins; ties go to the most recently active thread.
            scored = [(score, episode_rank.get(th.last_episode, -1), th) for th in series.threads
                      if (score := _link_score(t.kind, participants, th)) > 0]
            if scored:
                best_score, _, best = max(scored, key=lambda s: (s[0], s[1]))
                entry.link_score = round(best_score, 2)
                best.entries.append(entry)
                best.participants = sorted(set(best.participants) | set(participants))
                target = best
            else:
                target = SeriesThread(thread_id=f"S{len(series.threads) + 1:02d}", kind=t.kind,
                                      participants=participants, entries=[entry])
                series.threads.append(target)
            series.thread_map[f"{ep.episode_id}:{t.key}"] = target.thread_id

    # Canonical characters (speaker + verified names mentioned) and series thread IDs per event.
    for ep in ordered:
        events = {e.event_id: e for e in ep.events}
        for rec in consensus.get(ep.episode_id, []):
            event = events[rec.event_id]
            present = index.mentions(event.content)
            if event.speaker and (speaker := index.resolve(event.speaker)):
                present.add(speaker)
            rec.characters = sorted(present)
            rec.threads = sorted({series.thread_map.get(f"{ep.episode_id}:{key}", f"?{key}") for key in rec.threads})
    return series, index


def _link_score(kind: str, participants: list[str], thread: SeriesThread) -> float:
    """Heuristic link strength between an episode thread and an existing series thread.

    - Multi-character threads need at least two shared participants. Same kind: link.
      Different kind: link only if the cast is nearly identical (Jaccard >= 0.75), because a
      long story can change kind (betrayal -> revenge) while keeping its cast.
    - A one-person arc links only to a one-person arc of the same kind and person.
    Returns 0 for no link; otherwise the cast Jaccard (+0.25 for same kind) as link strength.
    """
    p, q = set(participants), set(thread.participants)
    if not p or not q:
        return 0.0
    shared = len(p & q)
    jaccard = shared / len(p | q)
    if len(p) == 1 or len(q) == 1:
        return 1.25 if p == q and kind == thread.kind else 0.0
    if shared < 2:
        return 0.0
    if kind == thread.kind:
        return jaccard + 0.25
    return jaccard if jaccard >= 0.75 else 0.0
