use anyhow::Result;
use tiktoken_rs::{CoreBPE, cl100k_base};

pub struct Tokens {
    tokenizer: CoreBPE,
    pub max_tokens: usize,
    tokens: Vec<u32>,
}

impl Tokens {
    pub fn new(content: &str, max_tokens: usize) -> Result<Self> {
        let tokenizer = cl100k_base()?;
        let tokens = tokenizer.encode_with_special_tokens(content);
        Ok(Self {
            tokenizer,
            max_tokens,
            tokens,
        })
    }

    pub fn exceed_max(&self) -> bool {
        if self.tokens.len() >= self.max_tokens {
            return true;
        }
        false
    }

    fn chunks(&self) -> impl Iterator<Item = &[u32]> {
        self.tokens.chunks(self.max_tokens)
    }

    pub fn decoded_chunks(&self) -> impl Iterator<Item = Result<String>> {
        self.chunks()
            .map(|chunk| self.tokenizer.decode(chunk.to_vec()))
    }
}
