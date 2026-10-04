//! Two long dialogues for checking the chat: 12 messages each, one in
//! Russian about deploying a new lesson, one in English about choosing a
//! memory strategy. Each sets its goal in the first message, adds
//! clarifications, constraints and a term along the way, asks follow-ups that
//! only make sense with what came before, goes off topic once, and ends by
//! asking for a recap.

use serde::Serialize;

#[derive(Serialize, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Slot {
    Goal,
    Clarified,
    Constraint,
    Term,
}

/// Something a message adds to the task state.
#[derive(Serialize, Debug)]
pub struct Note {
    pub slot: Slot,
    /// What a good state update would write.
    pub text: &'static str,
    /// The check: the state contains one of these after the turn (ignoring
    /// case), in whatever words the model chose.
    pub keys: &'static [&'static str],
}

#[derive(Serialize, Debug)]
pub struct Step {
    pub say: &'static str,
    /// A good self-contained search query for this message, in the
    /// documents' language: what the state update should produce. The tests'
    /// fake model returns it; the eval shows the real one.
    pub query: &'static str,
    pub notes: &'static [Note],
    /// The files a source of the answer should come from. Empty: the message
    /// is off topic, and the right answer is "I don't know".
    pub sources: &'static [&'static str],
    /// Facts the answer should state: each group is one fact, found if the
    /// answer contains any of its variants, ignoring case.
    pub facts: &'static [&'static [&'static str]],
}

#[derive(Serialize, Debug)]
pub struct Scenario {
    pub id: &'static str,
    pub title: &'static str,
    /// The goal is kept if, after every turn, it contains one of these.
    pub goal_keys: &'static [&'static str],
    pub steps: &'static [Step],
}

const DEPLOYMENT: &str = "DEPLOYMENT.md";
const AGENTS: &str = "AGENTS.md";
const L09: &str = "09. Context Compression/README.md";
const L10: &str = "10. Context Management Strategies/README.md";
const L11: &str = "11. Agent Memory Model/README.md";

pub const REFUSAL: &[&str] = &["don't know", "не знаю"];

pub const SCENARIOS: &[Scenario] = &[
    Scenario {
        id: "deploy",
        title: "Деплой нового урока (RU)",
        goal_keys: &["депло", "deploy", "развер", "выкат"],
        steps: &[
            Step {
                say: "Привет! Хочу добавить в репозиторий новый урок и задеплоить его на свой VDS через существующий пайплайн. С чего начать?",
                query: "How do I add a new lesson to the repository and get it deployed?",
                notes: &[Note {
                    slot: Slot::Goal,
                    text: "Добавить в репозиторий новый урок и задеплоить его на VDS через существующий пайплайн",
                    keys: &["депло", "deploy"],
                }],
                sources: &[AGENTS],
                facts: &[&["cargo.toml"]],
            },
            Step {
                say: "Уточню: урок будет веб-приложением на axum, номер 25. Он задеплоится автоматически?",
                query: "Which lessons deploy: does a web app lesson that depends on axum get deployed automatically?",
                notes: &[
                    Note {
                        slot: Slot::Clarified,
                        text: "Урок — веб-приложение на axum",
                        keys: &["axum"],
                    },
                    Note {
                        slot: Slot::Clarified,
                        text: "Номер урока — 25",
                        keys: &["25"],
                    },
                ],
                sources: &[DEPLOYMENT, AGENTS],
                facts: &[&["master", "push"]],
            },
            Step {
                say: "Ограничение: ставить Docker на VDS нельзя, и у деплой-ключа нет sudo. Как тогда устроен деплой?",
                query: "How is a lesson deployed without Docker and without sudo: static musl binary and a systemd user unit?",
                notes: &[
                    Note {
                        slot: Slot::Constraint,
                        text: "Без Docker на VDS",
                        keys: &["docker"],
                    },
                    Note {
                        slot: Slot::Constraint,
                        text: "У деплой-ключа нет sudo",
                        keys: &["sudo"],
                    },
                ],
                sources: &[DEPLOYMENT],
                facts: &[&["musl"], &["systemd"]],
            },
            Step {
                say: "На каком порту и под каким именем сервиса он поднимется?",
                query: "Which port and which systemd service name does a deployed lesson get on the VDS?",
                notes: &[],
                sources: &[DEPLOYMENT],
                facts: &[&["4025"], &["ai-advent-lesson-25"]],
            },
            Step {
                say: "Термин: «урок» — это папка вида `NN. Title` с отдельным Cargo-крейтом, дальше называй так. Общего workspace ведь нет?",
                query: "Each lesson folder named NN. Title is an independent Cargo crate, there is no workspace",
                notes: &[Note {
                    slot: Slot::Term,
                    text: "урок — папка вида `NN. Title` с отдельным Cargo-крейтом",
                    keys: &["nn. title", "крейт", "crate"],
                }],
                sources: &[AGENTS],
                facts: &[&["workspace"]],
            },
            Step {
                say: "Какие секреты и переменные GitHub нужны пайплайну, чтобы урок заработал?",
                query: "Which GitHub secrets and variables does the deploy pipeline need?",
                notes: &[],
                sources: &[DEPLOYMENT],
                facts: &[&["vds_ssh_key"], &["openai_api_key"]],
            },
            Step {
                say: "Кстати, какая погода будет завтра в Москве?",
                query: "What will the weather be in Moscow tomorrow?",
                notes: &[],
                sources: &[],
                facts: &[REFUSAL],
            },
            Step {
                say: "После деплоя сервис умирает, как только закрывается SSH-сессия. Что делать?",
                query: "What one-time command must be run on the VDS so the lesson services keep running after the SSH session closes?",
                notes: &[],
                sources: &[DEPLOYMENT],
                facts: &[&["enable-linger"]],
            },
            Step {
                say: "Будут ли cargo fmt и clippy блокировать сборку моего урока?",
                query: "Do cargo fmt and cargo clippy block the build-test job in CI?",
                notes: &[],
                sources: &[AGENTS],
                facts: &[&["continue-on-error", "не блок", "non-blocking"]],
            },
            Step {
                say: "А как потом передеплоить старый урок, например 06, не трогая новый?",
                query: "How do you redeploy an older lesson with a manual workflow_dispatch run?",
                notes: &[],
                sources: &[AGENTS, DEPLOYMENT],
                facts: &[&["workflow_dispatch", "run workflow"]],
            },
            Step {
                say: "Если в .env на VDS значение секрета оказалось пустым — в чём обычно причина?",
                query: "Why does a value in the lesson's .env show up empty after a deploy?",
                notes: &[],
                sources: &[DEPLOYMENT],
                facts: &[&["имен", "name", "назван"]],
            },
            Step {
                say: "Подведи итог: какая у нас цель, что я уже уточнил, какие ограничения и что делать по шагам?",
                query: "Adding a new lesson and deploying it on the VDS: steps, port, service name and secrets",
                notes: &[],
                sources: &[DEPLOYMENT, AGENTS],
                facts: &[&["4025"], &["docker"], &["axum"]],
            },
        ],
    },
    Scenario {
        id: "memory",
        title: "Choosing a memory strategy (EN)",
        goal_keys: &["context", "memory", "контекст", "памят"],
        steps: &[
            Step {
                say: "I'm building a customer-support chatbot that holds long conversations, and I want to borrow a context-management approach from these lessons. What options do they compare?",
                query: "Which context management strategies do the lessons compare for a long conversation?",
                notes: &[Note {
                    slot: Slot::Goal,
                    text: "Choose a context-management approach for a long-conversation customer-support chatbot, borrowing from the lessons",
                    keys: &["context", "memory"],
                }],
                sources: &[L10],
                facts: &[&["sliding"], &["sticky"], &["branch"]],
            },
            Step {
                say: "Constraint: I can afford at most one extra LLM call per turn. What does each option cost in calls?",
                query: "How many extra LLM calls per turn does each memory strategy cost?",
                notes: &[Note {
                    slot: Slot::Constraint,
                    text: "At most one extra LLM call per turn",
                    keys: &["one extra", "1 extra", "single extra", "at most one", "one additional"],
                }],
                sources: &[L10, L11],
                facts: &[&["sticky"]],
            },
            Step {
                say: "How does the summary-based compression decide when to fold old messages?",
                query: "When does context compression fold old messages into the summary?",
                notes: &[],
                sources: &[L09],
                facts: &[&["10"]],
            },
            Step {
                say: "Term: by \"memory\" I mean only what is injected into the prompt, not what is stored on disk. With that definition, how does memory shape each request?",
                query: "How does memory shape what gets sent to the model in each request?",
                notes: &[Note {
                    slot: Slot::Term,
                    text: "memory — only what is injected into the prompt, not what is stored on disk",
                    keys: &["injected", "prompt"],
                }],
                sources: &[L11],
                facts: &[&["long-term", "long_term"]],
            },
            Step {
                say: "Which approach still keeps the full raw transcript on disk anyway?",
                query: "Which approach keeps the full raw conversation history on disk?",
                notes: &[],
                sources: &[L09, L10],
                facts: &[&["history.json", "state.json"]],
            },
            Step {
                say: "What happens to a constraint mentioned early under the sliding window after many turns?",
                query: "What happens to an early constraint under the sliding window strategy after many turns?",
                notes: &[],
                sources: &[L10],
                facts: &[&["dropped", "gone", "forgot", "lost", "aged out"]],
            },
            Step {
                say: "Clarification: the bot must also remember each customer's preferences across sessions. Which lesson covers that?",
                query: "Long-term memory that survives across tasks and sessions about the user's preferences",
                notes: &[Note {
                    slot: Slot::Clarified,
                    text: "The bot must remember each customer's preferences across sessions",
                    keys: &["preference"],
                }],
                sources: &[L11],
                facts: &[&["long-term", "long_term"]],
            },
            Step {
                say: "Off-topic: what is the capital of Australia?",
                query: "What is the capital of Australia?",
                notes: &[],
                sources: &[],
                facts: &[REFUSAL],
            },
            Step {
                say: "How does the memory model decide whether a fact goes to working or long-term memory?",
                query: "How does the agent memory model decide whether a fact goes to working memory or long-term memory?",
                notes: &[],
                sources: &[L11],
                facts: &[&["prompt"]],
            },
            Step {
                say: "How many extra LLM calls per turn does that three-layer memory model cost?",
                query: "Token cost of the memory layers: how many extra LLM calls per turn?",
                notes: &[],
                sources: &[L11],
                facts: &[&["two-to-three", "two to three", "2-3", "2–3", "2 to 3"]],
            },
            Step {
                say: "Given my constraint, which approach fits best?",
                query: "Sticky facts strategy: one extra LLM call per turn to extract facts",
                notes: &[],
                sources: &[L10],
                facts: &[&["sticky"]],
            },
            Step {
                say: "Sum up: what is my goal, what have I clarified, and which constraints and terms are fixed?",
                query: "Context strategy for a long support chat: sticky facts, long-term memory and their cost",
                notes: &[],
                sources: &[L10, L11],
                facts: &[&["support"], &["one extra", "1 extra", "single extra", "at most one", "one additional"], &["preference"]],
            },
        ],
    },
];

pub fn find(id: &str) -> Option<&'static Scenario> {
    SCENARIOS.iter().find(|s| s.id == id)
}

impl Step {
    pub fn in_scope(&self) -> bool {
        !self.sources.is_empty()
    }
}
