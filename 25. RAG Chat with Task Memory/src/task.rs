//! The task state: what the dialogue is for, kept apart from the dialogue.
//!
//! The model only proposes a change: the goal, and the items the user's
//! latest message adds or withdraws. The merge is done here, by rules, so the
//! state can't lose the goal or an earlier constraint because a reply forgot
//! to repeat it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// At most this many items per list; the oldest goes first.
pub const MAX_ITEMS: usize = 20;
const MAX_ITEM_CHARS: usize = 300;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Term {
    pub term: String,
    pub meaning: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct TaskState {
    /// What the user wants to achieve in the whole dialogue.
    pub goal: String,
    /// The turn the goal was set at, or last changed at.
    pub goal_turn: usize,
    /// What the user has told about their situation or task.
    pub clarified: Vec<String>,
    /// What must or must not be done.
    pub constraints: Vec<String>,
    /// Words the user has given a meaning to.
    pub terms: Vec<Term>,
    /// The last turn that changed anything.
    pub updated_turn: usize,
}

/// What the model proposes after one user message.
#[derive(Debug, Default, PartialEq)]
pub struct Update {
    pub goal: String,
    pub goal_changed: bool,
    pub clarified: Vec<String>,
    pub constraints: Vec<String>,
    pub terms: Vec<Term>,
    /// Earlier items the user has withdrawn.
    pub remove: Vec<String>,
    /// The message as a self-contained search query.
    pub search_query: String,
}

/// What one turn changed, for the page and the eval.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(default)]
pub struct Change {
    /// The goal, when this turn set or changed it.
    pub goal: Option<String>,
    /// `constraint: …`, `clarified: …`, `term: X — meaning`.
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

impl Change {
    pub fn is_empty(&self) -> bool {
        self.goal.is_none() && self.added.is_empty() && self.removed.is_empty()
    }
}

/// Lower case, one space between words, no punctuation at the ends: two
/// items are the same item if this is the same.
fn norm(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

fn clean(text: &str) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.chars().count() <= MAX_ITEM_CHARS {
        return text;
    }
    let mut short: String = text.chars().take(MAX_ITEM_CHARS).collect();
    short.push('…');
    short
}

/// A string, or the first string field of an object (`{"text": "..."}`).
fn as_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => map.values().find_map(|v| v.as_str().map(str::to_string)),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn texts(v: &Value) -> Vec<String> {
    match v {
        Value::Array(items) => items.iter().filter_map(as_text).collect(),
        Value::String(s) if !s.trim().is_empty() => vec![s.clone()],
        _ => Vec::new(),
    }
}

/// `[{"term": ..., "meaning": ...}]`, or `{"term": "meaning"}`.
fn terms(v: &Value) -> Vec<Term> {
    match v {
        Value::Array(items) => items
            .iter()
            .filter_map(|t| {
                let term = t.get("term").and_then(as_text)?;
                let meaning = t
                    .get("meaning")
                    .or_else(|| t.get("definition"))
                    .and_then(as_text)
                    .unwrap_or_default();
                Some(Term { term, meaning })
            })
            .collect(),
        Value::Object(map) => map
            .iter()
            .filter_map(|(term, meaning)| {
                Some(Term {
                    term: term.clone(),
                    meaning: as_text(meaning)?,
                })
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// The JSON object in a reply, possibly wrapped in a code fence or prose.
pub fn json_object(text: &str) -> Option<&str> {
    match (text.find('{'), text.rfind('}')) {
        (Some(start), Some(end)) if start < end => Some(&text[start..=end]),
        _ => None,
    }
}

/// Reads the model's update leniently: a missing field is empty, a list may
/// hold strings or objects.
pub fn parse_update(text: &str) -> Result<Update, String> {
    let v: Value = json_object(text)
        .and_then(|j| serde_json::from_str(j).ok())
        .ok_or("the state update is not a JSON object")?;
    if !v.is_object() {
        return Err("the state update is not a JSON object".to_string());
    }
    let s = |key: &str| v.get(key).and_then(as_text).unwrap_or_default().trim().to_string();
    Ok(Update {
        goal: s("goal"),
        goal_changed: match v.get("goal_changed") {
            Some(Value::Bool(b)) => *b,
            Some(Value::String(s)) => s.eq_ignore_ascii_case("true"),
            _ => false,
        },
        clarified: v.get("clarified").map(texts).unwrap_or_default(),
        constraints: v.get("constraints").map(texts).unwrap_or_default(),
        terms: v.get("terms").map(terms).unwrap_or_default(),
        remove: v.get("remove").map(texts).unwrap_or_default(),
        search_query: s("search_query"),
    })
}

fn add_item(list: &mut Vec<String>, item: &str, label: &str, change: &mut Change) {
    let item = clean(item);
    let key = norm(&item);
    if key.is_empty() || list.iter().any(|x| norm(x) == key) {
        return;
    }
    if list.len() >= MAX_ITEMS {
        list.remove(0);
    }
    change.added.push(format!("{label}: {item}"));
    list.push(item);
}

/// An item matches a removal when it's the same text, or contains the
/// removal's text (at least 8 characters, so "a" removes nothing).
fn matches_removal(item: &str, key: &str) -> bool {
    let item = norm(item);
    item == key || (key.chars().count() >= 8 && item.contains(key))
}

impl TaskState {
    pub fn is_empty(&self) -> bool {
        self.goal.is_empty()
            && self.clarified.is_empty()
            && self.constraints.is_empty()
            && self.terms.is_empty()
    }

    /// Merges an update. The goal is set once and changes only when the
    /// update says so; lists only grow, except by an explicit removal.
    pub fn apply(&mut self, update: &Update, turn: usize) -> Change {
        let mut change = Change::default();

        // Removals first, so "not X but Y" can replace X in one turn.
        for r in &update.remove {
            let key = norm(r);
            if key.is_empty() {
                continue;
            }
            for (label, list) in [
                ("clarified", &mut self.clarified),
                ("constraint", &mut self.constraints),
            ] {
                list.retain(|item| {
                    let hit = matches_removal(item, &key);
                    if hit {
                        change.removed.push(format!("{label}: {item}"));
                    }
                    !hit
                });
            }
            self.terms.retain(|t| {
                let hit = norm(&t.term) == key;
                if hit {
                    change.removed.push(format!("term: {}", t.term));
                }
                !hit
            });
        }

        let goal = clean(&update.goal);
        if !goal.is_empty()
            && (self.goal.is_empty() || (update.goal_changed && norm(&goal) != norm(&self.goal)))
        {
            self.goal = goal.clone();
            self.goal_turn = turn;
            change.goal = Some(goal);
        }

        for item in &update.clarified {
            add_item(&mut self.clarified, item, "clarified", &mut change);
        }
        for item in &update.constraints {
            add_item(&mut self.constraints, item, "constraint", &mut change);
        }
        for t in &update.terms {
            let term = clean(&t.term);
            let meaning = clean(&t.meaning);
            if norm(&term).is_empty() {
                continue;
            }
            match self.terms.iter_mut().find(|x| norm(&x.term) == norm(&term)) {
                Some(x) if norm(&x.meaning) == norm(&meaning) || meaning.is_empty() => {}
                Some(x) => {
                    x.meaning = meaning.clone();
                    change.added.push(format!("term: {term} — {meaning}"));
                }
                None => {
                    if self.terms.len() >= MAX_ITEMS {
                        self.terms.remove(0);
                    }
                    change.added.push(format!("term: {term} — {meaning}"));
                    self.terms.push(Term { term, meaning });
                }
            }
        }

        if !change.is_empty() {
            self.updated_turn = turn;
        }
        change
    }

    /// The state as a block of the system prompt.
    pub fn render(&self) -> String {
        let list = |items: &[String]| {
            if items.is_empty() {
                "- (none yet)".to_string()
            } else {
                items.iter().map(|i| format!("- {i}")).collect::<Vec<_>>().join("\n")
            }
        };
        let terms = if self.terms.is_empty() {
            "- (none yet)".to_string()
        } else {
            self.terms
                .iter()
                .map(|t| format!("- {}: {}", t.term, t.meaning))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let goal = if self.goal.is_empty() { "(not set yet)" } else { &self.goal };
        format!(
            "TASK STATE (kept by the system for the whole dialogue; it outranks older messages)\n\
Goal of the dialogue: {goal}\n\
Clarified by the user:\n{}\n\
Constraints:\n{}\n\
Terms (use them in exactly this meaning):\n{terms}",
            list(&self.clarified),
            list(&self.constraints),
        )
    }

    /// Every word of the state, for checks that look for a keyword anywhere.
    pub fn all_text(&self) -> String {
        let mut out = vec![self.goal.clone()];
        out.extend(self.clarified.iter().cloned());
        out.extend(self.constraints.iter().cloned());
        out.extend(self.terms.iter().map(|t| format!("{}: {}", t.term, t.meaning)));
        out.join("\n").to_lowercase()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn update(v: Value) -> Update {
        parse_update(&v.to_string()).unwrap()
    }

    #[test]
    fn the_goal_is_set_once_and_changes_only_on_request() {
        let mut s = TaskState::default();
        let c = s.apply(&update(json!({ "goal": "Deploy lesson 25" })), 1);
        assert_eq!(c.goal.as_deref(), Some("Deploy lesson 25"));
        assert_eq!((s.goal.as_str(), s.goal_turn, s.updated_turn), ("Deploy lesson 25", 1, 1));

        // A reworded goal without goal_changed, or an empty one, is ignored.
        let c = s.apply(&update(json!({ "goal": "Weather in Moscow" })), 2);
        assert!(c.is_empty() && s.goal == "Deploy lesson 25" && s.updated_turn == 1, "{c:?}");
        s.apply(&update(json!({ "goal": "", "goal_changed": true })), 3);
        assert_eq!(s.goal, "Deploy lesson 25");

        let c = s.apply(&update(json!({ "goal": "Deploy lesson 26", "goal_changed": "true" })), 4);
        assert_eq!(c.goal.as_deref(), Some("Deploy lesson 26"));
        assert_eq!((s.goal_turn, s.updated_turn), (4, 4));
    }

    #[test]
    fn lists_grow_without_repeats_and_shrink_only_by_removal() {
        let mut s = TaskState::default();
        let c = s.apply(
            &update(json!({
                "clarified": ["Lesson 25 is an axum web app"],
                "constraints": ["No Docker on the VDS", { "text": "No sudo" }],
                "terms": [{ "term": "lesson", "meaning": "an NN. Title folder" }],
            })),
            1,
        );
        assert_eq!(c.added.len(), 4, "{c:?}");
        assert_eq!(s.constraints, vec!["No Docker on the VDS", "No sudo"]);

        // Repeats (any case, spacing, end punctuation) aren't added again;
        // a list the model leaves out keeps what it had.
        let c = s.apply(
            &update(json!({ "constraints": ["no  docker on the VDS."], "terms": { "Lesson": "an NN. Title folder" } })),
            2,
        );
        assert!(c.is_empty(), "{c:?}");
        assert_eq!(s.constraints.len(), 2);
        assert_eq!(s.clarified.len(), 1);

        // A term gets a new meaning; a withdrawn constraint goes, and its
        // replacement comes in the same turn.
        let c = s.apply(
            &update(json!({
                "terms": [{ "term": "lesson", "meaning": "a Cargo crate" }],
                "remove": ["No sudo"],
                "constraints": ["sudo is available"],
            })),
            3,
        );
        assert_eq!(c.removed, vec!["constraint: No sudo"]);
        assert_eq!(s.constraints, vec!["No Docker on the VDS", "sudo is available"]);
        assert_eq!(s.terms[0].meaning, "a Cargo crate");
        assert!(c.added.iter().any(|a| a.starts_with("term: lesson")));

        // A short removal text removes nothing by accident.
        s.apply(&update(json!({ "remove": ["no"] })), 4);
        assert_eq!(s.constraints.len(), 2);

        let r = s.render();
        assert!(r.contains("Goal of the dialogue: (not set yet)"));
        assert!(r.contains("- No Docker on the VDS") && r.contains("- lesson: a Cargo crate"));
        assert!(s.all_text().contains("docker"));
    }

    #[test]
    fn lists_are_capped_and_replies_are_read_leniently() {
        let mut s = TaskState::default();
        for i in 0..MAX_ITEMS + 3 {
            s.apply(&update(json!({ "clarified": [format!("detail number {i}")] })), i + 1);
        }
        assert_eq!(s.clarified.len(), MAX_ITEMS);
        assert_eq!(s.clarified[0], "detail number 3", "the oldest go first");

        let u = parse_update("```json\n{\"search_query\": \" port of lesson 25 \", \"constraints\": \"no Docker\"}\n```").unwrap();
        assert_eq!(u.search_query, "port of lesson 25");
        assert_eq!(u.constraints, vec!["no Docker"]);
        assert!(!u.goal_changed && u.goal.is_empty());
        assert!(parse_update("sorry, no JSON").is_err());
        assert!(parse_update("[1, 2]").is_err());
    }
}
