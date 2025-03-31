#Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import torch
import logging, warnings
import string
import typing as tp
import gc

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..training.utils import copy_state_dict
from .utils import load_ckpt_state_dict

from torch import nn

from .Qformer import BertConfig, BertLMHeadModel, BertAttention, BertIntermediate, BertOutput
from transformers import BertTokenizer

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class IntConditioner(Conditioner):
    def __init__(self, 
                output_dim: int,
                min_val: int=0,
                max_val: int=512
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
            
            #self.int_embedder.to(device)
    
            ints = torch.tensor(ints).to(device)
            ints = ints.clamp(self.min_val, self.max_val)
    
            int_embeds = self.int_embedder(ints).unsqueeze(1)
    
            return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]

class NumberConditioner(Conditioner):
    '''
        Conditioner that takes a list of floats, normalizes them for a given range, and returns a list of embeddings
    '''
    def __init__(self, 
                output_dim: int,
                min_val: float=0,
                max_val: float=1
                ):
        super().__init__(output_dim, output_dim)

        self.min_val = min_val
        self.max_val = max_val

        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
    
            # Cast the inputs to floats
            floats = [float(x) for x in floats]

            floats = torch.tensor(floats).to(device)

            floats = floats.clamp(self.min_val, self.max_val)
    
            normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)

            # Cast floats to same type as embedder
            embedder_dtype = next(self.embedder.parameters()).dtype
            normalized_floats = normalized_floats.to(embedder_dtype)

            float_embeds = self.embedder(normalized_floats).unsqueeze(1)
    
            return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]

class CLAPTextConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 use_text_features = False,
                 feature_layer_ix: int = -1,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False,
                 finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)

        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch

        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True
        )["hidden_states"][layer_ix]

        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)

        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features([texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(texts, layer_ix=self.feature_layer_ix, device=device)
            return [self.proj_out(text_features), text_attention_mask]

        # Fix for CLAP bug when only one text is passed
        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)

        text_embedding = text_embedding.unsqueeze(1).to(device)

        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]

class CLAPAudioConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 clap_ckpt_path,
                 audio_model_type="HTSAT-base", 
                 enable_fusion=True,
                 project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')

                if self.finetune:
                    self.model = model
                else: 
                    self.__dict__["model"] = model

                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)

                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()

            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch

        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]] , device: tp.Any = "cuda") -> tp.Any:

        self.model.to(device)

        if isinstance(audios, list) or isinstance(audios, tuple):
            audios = torch.cat(audios, dim=0)

        # Convert to mono
        mono_audios = audios.mean(dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)

        audio_embedding = audio_embedding.unsqueeze(1).to(device)

        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]
    
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    
    return self

# Define BertLayer for cross attention

from .Qformer import BertConfig, BertLMHeadModel, BertAttention, BertIntermediate, BertOutput
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)

        self.crossattention = BertAttention(
            config, is_cross_attention=True
        )
        self.has_cross_attention = True

        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        self.intermediate_query = BertIntermediate(config)
        self.output_query = BertOutput(config)
    
    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        query_length=0,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None

        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )

        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:-1]
        present_key_value = self_attention_outputs[-1]

        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions=output_attentions,
        )

        outputs = (
            outputs + cross_attention_outputs[1:-1]
        )  # add cross attentions if we output attention weights


        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs

        outputs = outputs + (present_key_value,)

        return outputs


class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
              "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
              "google/flan-t5-xl", "google/flan-t5-xxl"]
    
    T5_MODEL_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "t5-xl": 2048,
        "t5-xxl": 4096,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048,
        "google/flan-t5-xxl": 4096,
    }


    def init_Qformer(cls, num_query_token, vision_width, freeze, cross_attention_freq=2):

        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel.from_pretrained(
            "bert-base-uncased", config=encoder_config
        )

        qformer_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="right")
        qformer_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        Qformer.resize_token_embeddings(len(qformer_tokenizer))

        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        # optional, if not loading weights
        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            print("freeze Qformer")

        return Qformer, query_tokens

    def __init__(
            self,
            output_dim: int,
            t5_model_name: str = "t5-base",
            max_length: str = 128,
            enable_grad: bool = False,
            project_out: bool = False
    ):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)
        
        from transformers import T5EncoderModel, AutoTokenizer, AutoModel

        from peft import (
            LoraConfig,
            get_peft_model,
            get_peft_model_state_dict,
            set_peft_model_state_dict,
        )

        self.qformer_proj_norm = nn.LayerNorm(768)
        self.audio_proj_norm_qformer = nn.LayerNorm(768, elementwise_affine=False)
        # Cross attention layer for qformer and t5
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.encoder_width = 768
        self.cross_attend = BertLayer(bert_config)
        # self.proj_out_cross = nn.Linear(1024,768)

        self.max_length = max_length
        self.enable_grad = enable_grad

        # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length = max_length)
                # model = T5EncoderModel.from_pretrained(t5_model_name, max_length=max_length).train(enable_grad).requires_grad_(enable_grad)
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                ckpt = torch.load('/fs/nexus-projects/brain_project/try_t5.pt')
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
                model.load_state_dict(ckpt,strict=True)

                self.llm_model = AutoModel.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B",
                    load_in_8bit=False,
                    # torch_dtype=torch.float16,
                    device_map="auto",
                )

                self.llm_model_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
                self.num_new_tokens = 64
                self.IGNORE_TOKEN_ID=-100

                # 1. LLM has the special token "<ad>" for system message to generate image -> add_tokens "<img>" -> 32000
                self.llm_model_tokenizer.add_tokens(["<ad>"], special_tokens=False)

                # 2. LLM contains 64 tokens to summarize image and text information for conversation system -> add_tokens "<img_0>...<img_63>" -> 32001~32064
                new_token_list = [f"<ad_{i}>" for i in range(self.num_new_tokens)]
                self.llm_model_tokenizer.add_tokens(new_token_list, special_tokens=False)

                # 3. count new tokens and resize tokenizer
                self.num_new_tokens = self.num_new_tokens + 1
                self.llm_model.resize_token_embeddings(len(self.llm_model_tokenizer))
                self.llm_model_tokenizer.ad_start_token_id = self.llm_model_tokenizer.convert_tokens_to_ids("<ad_0>")
                # ------------------- #
                # build new lm head
                self.lm_head = nn.Linear(4096, len(self.llm_model_tokenizer), bias=False)
                # initialize a new variable to store vocab_size
                self.vocab_size = len(self.llm_model_tokenizer)

                # 4. Initialize the new embeddings with original embeddings
                input_embeddings = self.llm_model.model.embed_tokens.weight.data
                output_embeddings = self.llm_model.lm_head.weight.data
                self.original_LLM_word_embedding_0 = input_embeddings[0]
                self.original_LLM_language_model_head_0 = output_embeddings[0]
                # ------------- #
                input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
                # ------------- #
                input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
                output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
                # ------------- #
                self.llm_model.model.embed_tokens.weight.data = input_embeddings
                self.lm_head.weight.data = output_embeddings

                # 5. Initialize the Qformer
                self.Qformer, self.query_tokens = self.init_Qformer(32, 768, True)
                self.llm_to_qformer_projection = nn.Linear(4096,768)

                # Add lora modules to the model
                config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type='CAUSAL_LM')
                self.llm_model = get_peft_model(self.llm_model, config)

            finally:
                logging.disable(previous_level)
   
        if self.enable_grad:
            self.model = model
        else: 
            self.__dict__["model"] = model


    def forward(self, texts: tp.List[str],  device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        llm_input_ids = []
        llm_input_attention_mask = []
        llm_targets_list = []
        qformer_input_attention_mask = []

        for text in texts:
            # define a system prompt for the LLM
            llm_caption_system = "A chat between a curious user and an artificial intelligence assistant. The assistant can generate <ad>. "

            # construct the prompt for the LLM
            llm_caption_interim = "Please generate an audio for the following caption: " + text
            llm_caption_last = " Here is the audio for the given caption: [ad]"

            append_str = ""
            for i in range(self.num_new_tokens - 1):
                append_str += f" <ad_{i}>"
            llm_caption = llm_caption_last.replace(" [ad]", append_str)

            # add the system prompt to the LLM prompt
            llm_caption = llm_caption_system + llm_caption_interim + llm_caption_last

            # tokenize the prompt

            input_ids_max_len = 512
            llm_caption_input_ids = self.llm_model_tokenizer(
                llm_caption,
                return_tensors="pt",
                padding="max_length",
                max_length=input_ids_max_len,
                truncation=True,
            ).input_ids[0]

            # generate LLM targets
            llm_targets = llm_caption_input_ids.clone()
            llm_targets[:1] = self.IGNORE_TOKEN_ID
            total_padding_len = int(llm_targets.ne(self.llm_model_tokenizer.pad_token_id).sum())

            instruction_len = len(
                self.llm_model_tokenizer(
                    llm_caption_system + llm_caption_interim,
                    max_length=input_ids_max_len,
                    truncation=True,
                ).input_ids) - 2

            llm_targets[1:(1 + instruction_len)] = self.IGNORE_TOKEN_ID
            llm_targets[total_padding_len:] = self.IGNORE_TOKEN_ID

            # append all
            llm_input_ids.append(llm_caption_input_ids)
            llm_input_attention_mask.append(llm_caption_input_ids.ne(self.llm_model_tokenizer.pad_token_id))
            llm_targets_list.append(llm_targets)
            qformer_input_attention_mask.append(llm_caption_input_ids.ge(self.llm_model_tokenizer.ad_start_token_id))


        llm_input_ids = torch.stack([torch.tensor(input_id) for input_id in llm_input_ids]).to(device)
        llm_input_attention_mask = torch.stack([torch.tensor(attention_mask) for attention_mask in llm_input_attention_mask]).to(device)
        llm_targets = torch.stack([torch.tensor(llm_target) for llm_target in llm_targets_list]).to(device)
        qformer_input_attention_mask = torch.stack([torch.tensor(qformer_attention_mask) for qformer_attention_mask in qformer_input_attention_mask]).to(device)

        # LLM Model
        llm_outputs = self.llm_model.model(
            input_ids=llm_input_ids,
            attention_mask=llm_input_attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        hidden_states_llm = llm_outputs[0]
        shift_labels = llm_targets[..., 1:].contiguous()

        # 5. Next token prediction language model loss: Enable model parallelism
        hidden_states = hidden_states_llm.to(torch.float32)
        logits = self.lm_head(hidden_states)
        shift_logits = logits[..., :-1, :].contiguous()
        # Flatten the tokens
        ce_loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        LM_loss = ce_loss_fct(shift_logits, shift_labels) * 1.0 #self.config.llm_loss_weight

        # for qformer
        hidden_states_llm = self.llm_to_qformer_projection(hidden_states_llm[:, :512, :]) # cut it to 512 max
        audio_input_for_qformer = self.qformer_proj_norm(hidden_states_llm)
        # audio_atts = torch.ones(audio_input_for_qformer.size()[:-1], dtype=torch.long).to(device) # can and should we convert the attention to pay attention only to the non padded tokens
        query_tokens = self.query_tokens.expand(audio_input_for_qformer.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=audio_input_for_qformer,
            encoder_attention_mask=qformer_input_attention_mask,
            return_dict=True,
        )
        query_output = self.audio_proj_norm_qformer(query_output.last_hidden_state)

        # T5 model

        self.model.to(device)
        self.proj_out.to(device)
        # self.proj_out_cross.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
            
        with torch.cuda.amp.autocast(dtype=torch.float16) and torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )["last_hidden_state"]
            
        embeddings = self.proj_out(embeddings.float())

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        # -----------------#

        # cross attention between qformer and t5
        qformer_attns = torch.ones(query_output.size()[:-1], dtype=torch.long).to(device)
        qformer_attns = self.get_bert_extended_attention_mask(qformer_attns, query_output.size()[:-1], device, False)

        embeddings = self.cross_attend(query_output,attention_mask=qformer_attns,encoder_hidden_states=embeddings,encoder_attention_mask=attention_mask)

        return embeddings, attention_mask, LM_loss
    
class PhonemeConditioner(Conditioner):
    """
    A conditioner that turns text into phonemes and embeds them using a lookup table
    Only works for English text

    Args:
        output_dim: the dimension of the output embeddings
        max_length: the maximum number of phonemes to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from g2p_en import G2p

        self.max_length = max_length

        self.g2p = G2p()

        # Reserving 0 for padding, 1 for ignored
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts] # shape [batch_size, length]
        
        phoneme_ignore = [" ", *string.punctuation]

        # Remove ignored phonemes and cut to max length
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]

        # Convert to ids
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]

        #Pad to match longest and make a mask tensor for the padding
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        
        phoneme_ids = torch.tensor(phoneme_ids).to(device)

        # Convert to embeddings
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        
        phoneme_embeds = self.proj_out(phoneme_embeds)

        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)
  
class TokenizerLUTConditioner(Conditioner):
    """
    A conditioner that embeds text using a lookup table on a pretrained tokenizer's vocabulary

    Args:
        tokenizer_name: the name of the tokenizer from the Hugging Face transformers library
        output_dim: the dimension of the output embeddings
        max_length: the maximum length of the text to embed
        project_out: whether to add another linear projection to the output embeddings
    """

    def __init__(
            self,
            tokenizer_name: str, # Name of a tokenizer from the Hugging Face transformers library
            output_dim: int,
            max_length: int = 1024,
            project_out: bool = False,
    ):
        super().__init__(output_dim, output_dim, project_out=project_out)
        
        from transformers import AutoTokenizer

         # Suppress logging from transformers
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        self.max_length = max_length

        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
    
        embeddings = self.token_embedder(input_ids)
            
        embeddings = self.proj_out(embeddings)

        embeddings = embeddings * attention_mask.unsqueeze(-1).float()

        return embeddings, attention_mask

class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)

        self.pretransform = pretransform

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.cat(audio, dim=0)

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        
        latents = self.pretransform.encode(audio)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]
                    
                else:
                    conditioner_input = x[condition_key]
                
                # if isinstance(conditioner_input, dict):
                #     if len(conditioner_inputs) == 0:
                #         conditioner_inputs = {C:[] for C in conditioner_input}
                #     for cond in conditioner_input:
                #         conditioner_inputs[cond].append(conditioner_input[cond])
                # else:
                conditioner_inputs.append(conditioner_input)
            
            output[key] = conditioner(conditioner_inputs, device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}
        
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))

            conditioners[id] = PretransformConditioner(pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)