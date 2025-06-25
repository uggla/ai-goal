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
