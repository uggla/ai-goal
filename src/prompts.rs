use ollama_rs::generation::chat::ChatMessage;

use crate::{Lang, OllamaAction};

pub trait OllamaPromptProvider {
    fn get_partial_prompt(&mut self, content: &str) -> Vec<ChatMessage>;
    fn get_final_prompt(&mut self, content: &str) -> Vec<ChatMessage>;
    fn get_action(&self) -> OllamaAction;
    fn get_lang(&self) -> Lang;
}

pub struct SummaryPrompt {
    action: OllamaAction,
    lang: Lang,
    partial_prompt: Vec<ChatMessage>,
    final_prompt: Vec<ChatMessage>,
}

impl SummaryPrompt {
    pub fn new(lang: Lang) -> Self {
        match lang {
            Lang::Fr => {
                let partial_prompt = vec![
                    ChatMessage::system(
                        "Tu es un assistant qui résume un texte dans sa langue d'origine et de manière consise.".to_string(),
                    ),
                    ChatMessage::user("Voici un extrait de texte à résumer :\n{{content}}".to_string()),
                ];

                let final_prompt = vec![
                    ChatMessage::system("Tu es un assistant de résumé.".to_string()),
                    ChatMessage::user(
                        "Voici plusieurs résumés partiels :\n{{content}}\nFais un résumé global."
                            .to_string(),
                    ),
                ];

                Self {
                    action: OllamaAction::Summary,
                    lang,
                    partial_prompt,
                    final_prompt,
                }
            }
            Lang::En => {
                let partial_prompt = vec![
                    ChatMessage::system(
                        "You are an assistant that summarizes a text in its original language and in a concise manner.".to_string(),
                    ),
                    ChatMessage::user("Here is an excerpt of text to summarize:\n{{content}}".to_string()),
                ];

                let final_prompt = vec![
                    ChatMessage::system("You are a summarization assistant.".to_string()),
                    ChatMessage::user(
                        "Here are several partial summaries:\n{{content}}\nPlease provide an overall summary."
                            .to_string(),
                    ),
                ];

                Self {
                    action: OllamaAction::Summary,
                    lang,
                    partial_prompt,
                    final_prompt,
                }
            }
        }
    }
}

impl OllamaPromptProvider for SummaryPrompt {
    fn get_partial_prompt(&mut self, content: &str) -> Vec<ChatMessage> {
        replace_chatmessage_content(&mut self.partial_prompt, content);
        self.partial_prompt.clone()
    }

    fn get_final_prompt(&mut self, content: &str) -> Vec<ChatMessage> {
        replace_chatmessage_content(&mut self.final_prompt, content);
        self.final_prompt.clone()
    }

    fn get_action(&self) -> OllamaAction {
        self.action
    }

    fn get_lang(&self) -> Lang {
        self.lang
    }
}

pub struct CreateChapterPrompt {
    action: OllamaAction,
    lang: Lang,
    partial_prompt: Vec<ChatMessage>,
    final_prompt: Vec<ChatMessage>,
}

impl CreateChapterPrompt {
    pub fn new(lang: Lang) -> Self {
        let system_prompt_fr = ChatMessage::system("
Tu es un assistant chargé d'analyser une transcription audio segmentée en blocs, avec des intervalles temporels explicites.

Chaque segment est de la forme :
[hh:mm:ss - hh:mm:ss] texte

Le traitement se déroule en deux étapes :
1. Pour chaque bloc reçu, tu dois extraire les idées principales et les reformuler si nécessaire. Ne produis pas de chapitrage global à ce stade.
2. Une fois tous les blocs reçus, tu recevras une instruction claire pour générer la liste complète des chapitres.

Les chapitres devront inclure :
- Un titre clair et concis
- Le timestamp de début du chapitre (hh:mm:ss), basé sur les segments analysés

Ne fais pas d’introduction ni de conclusion dans ta réponse. Ne fais pas de résumé global tant que l’instruction finale n’a pas été donnée."
        .to_string());
        let system_prompt_en = ChatMessage::system("
You are an assistant tasked with analyzing a segmented audio transcription with explicit time intervals.

Each segment is in the form:
[hh:mm:ss - hh:mm:ss] text

The processing occurs in two steps:
1. For each received block, you must extract the main ideas and rephrase them if necessary. Do not produce a global chaptering at this stage.
2. Once all blocks have been received, you will receive a clear instruction to generate the complete list of chapters.

The chapters must include:
- A clear and concise title
- The starting timestamp of the chapter (hh:mm:ss), based on the analyzed segments

Do not include an introduction or conclusion in your response. Do not provide a global summary until the final instruction has been given."
        .to_string());
        match lang {
            Lang::Fr => {
                let partial_prompt = vec![
                    system_prompt_fr.clone(),
                    ChatMessage::user(
                        "Voici un nouveau bloc de transcription à analyser:\n{{content}}"
                            .to_string(),
                    ),
                ];

                let final_prompt = vec![
                    system_prompt_fr.clone(),
                    ChatMessage::user("
Ceci est la seconde étape.
Tu peux maintenant générer la liste complète des chapitres à partir de l’ensemble des segments analysés précédemment que je vais te donner.

Pour chaque chapitre, indique :
- Un titre
- Le timestamp de début (hh:mm:ss)

Présente uniquement la liste, sans autre commentaire.
Voici le contenu:\n{{content}}"
                    .to_string(),
                    ),
                ];

                Self {
                    action: OllamaAction::Summary,
                    lang,
                    partial_prompt,
                    final_prompt,
                }
            }
            Lang::En => {
                let partial_prompt = vec![
                    system_prompt_en.clone(),
                    ChatMessage::user(
                        "Here is a new block of transcription to analyze:\n{{content}}".to_string(),
                    ),
                ];

                let final_prompt = vec![
                    system_prompt_en.clone(),
                    ChatMessage::user("
This is the second step.
You can now generate the complete list of chapters based on all the previously analyzed segments that I will give you.

For each chapter, indicate:
- A title
- The starting timestamp of the chapter (hh:mm:ss)

Present only the list, without any other comments.
Here is the content:\n{{content}}"
                    .to_string(),
                    ),
                ];

                Self {
                    action: OllamaAction::CreateChapters,
                    lang,
                    partial_prompt,
                    final_prompt,
                }
            }
        }
    }
}

impl OllamaPromptProvider for CreateChapterPrompt {
    fn get_partial_prompt(&mut self, content: &str) -> Vec<ChatMessage> {
        replace_chatmessage_content(&mut self.partial_prompt, content);
        self.partial_prompt.clone()
    }

    fn get_final_prompt(&mut self, content: &str) -> Vec<ChatMessage> {
        replace_chatmessage_content(&mut self.final_prompt, content);
        self.final_prompt.clone()
    }

    fn get_action(&self) -> OllamaAction {
        self.action
    }

    fn get_lang(&self) -> Lang {
        self.lang
    }
}

fn replace_chatmessage_content(prompt: &mut [ChatMessage], content: &str) {
    prompt.iter_mut().for_each(|o| {
        let new_content = o.content.replace("{{content}}", content);
        o.content = new_content;
    });
}

#[cfg(test)]
mod tests {
    use ollama_rs::generation::chat::MessageRole;

    use super::*;

    #[test]
    fn test_get_partial_prompt_fr() {
        let lang = Lang::Fr;
        let mut summary_prompt = SummaryPrompt::new(lang);
        let content = "This is a test content.";

        let partial_prompt = summary_prompt.get_partial_prompt(content);

        assert_eq!(partial_prompt.len(), 2);
        assert_eq!(
            partial_prompt[0].content,
            "Tu es un assistant qui résume un texte dans sa langue d'origine et de manière consise."
        );
        assert_eq!(partial_prompt[0].role, MessageRole::System);
        assert_eq!(
            partial_prompt[1].content,
            "Voici un extrait de texte à résumer :\nThis is a test content."
        );
        assert_eq!(partial_prompt[1].role, MessageRole::User);
    }

    #[test]
    fn test_get_final_prompt_fr() {
        let lang = Lang::Fr;
        let mut summary_prompt = SummaryPrompt::new(lang);
        let content = "This is a test content.";

        let final_prompt = summary_prompt.get_final_prompt(content);

        assert_eq!(final_prompt.len(), 2);
        assert_eq!(final_prompt[0].content, "Tu es un assistant de résumé.");
        assert_eq!(final_prompt[0].role, MessageRole::System);
        assert_eq!(
            final_prompt[1].content,
            "Voici plusieurs résumés partiels :\nThis is a test content.\nFais un résumé global."
        );
        assert_eq!(final_prompt[1].role, MessageRole::User);
    }
    #[test]
    fn test_get_partial_prompt_en() {
        let lang = Lang::En;
        let mut summary_prompt = SummaryPrompt::new(lang);
        let content = "This is a test content.";

        let partial_prompt = summary_prompt.get_partial_prompt(content);

        assert_eq!(partial_prompt.len(), 2);
        assert_eq!(
            partial_prompt[0].content,
            "You are an assistant that summarizes a text in its original language and in a concise manner."
        );
        assert_eq!(partial_prompt[0].role, MessageRole::System);
        assert_eq!(
            partial_prompt[1].content,
            "Here is an excerpt of text to summarize:\nThis is a test content."
        );
        assert_eq!(partial_prompt[1].role, MessageRole::User);
    }

    #[test]
    fn test_get_final_prompt_en() {
        let lang = Lang::En;
        let mut summary_prompt = SummaryPrompt::new(lang);
        let content = "This is a test content.";

        let final_prompt = summary_prompt.get_final_prompt(content);

        assert_eq!(final_prompt.len(), 2);
        assert_eq!(
            final_prompt[0].content,
            "You are a summarization assistant."
        );
        assert_eq!(final_prompt[0].role, MessageRole::System);
        assert_eq!(
            final_prompt[1].content,
            "Here are several partial summaries:\nThis is a test content.\nPlease provide an overall summary."
        );
        assert_eq!(final_prompt[1].role, MessageRole::User);
    }
}
