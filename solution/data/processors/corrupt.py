import torch
import math
import re


def permute_sentences(examples, data_args, p=1.0):
    """
    Permutate sentences

    Args:
        examples (Dict[Any]): DatasetDict
        data_args (DataArguments): data arguments
        p (float, optional): Permutation ratio. Defaults to 1.0.

    Returns:
        Dict[Any]: DatasetDict
    """
    p = data_args.permute_sentence_ratio

    answers = []
    context = []
    document_id = []
    ids = []
    question = []
    title = []
    no_batch = type(examples['context']) == str

    # no batch
    if no_batch == True:
        sentence_list = examples['context'].split('#')
        result = sentence_list.copy()

        num_sentences = len(sentence_list)
        num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
        substitutions = torch.randperm(num_sentences)[:num_to_permute]
        ordering = torch.arange(0, num_sentences)
        ordering[substitutions] = substitutions[torch.randperm(num_to_permute)]

        result = ' '.join([sentence_list[ordering[j]]
                          for j in range(len(sentence_list))])
        index = result.find('[ANSWER]')
        result = re.sub('\[ANSWER\]', '', result)

        answer = examples['answers']
        answer['answer_start'][0] = index

        return {'answers': answer,
                'context': result,
                'document_id': examples['document_id'],
                'id': examples['id'],
                'question': examples['question'],
                'title': examples['title']}

    # batch
    else:
        for i in range(len(examples['context'])):
            sentence_list = examples['context'][i].split('#')
            result = sentence_list.copy()

            num_sentences = len(sentence_list)
            num_to_permute = math.ceil((num_sentences * 2 * p) / 2.0)
            substitutions = torch.randperm(num_sentences)[:num_to_permute]
            ordering = torch.arange(0, num_sentences)
            ordering[substitutions] = substitutions[torch.randperm(
                num_to_permute)]

            result = ' '.join([sentence_list[ordering[j]]
                              for j in range(len(sentence_list))])
            index = result.find('[ANSWER]')
            result = re.sub('\[ANSWER\]', '', result)

            answer = examples['answers'][i]
            answer['answer_start'][0] = index

            answers.append(answer)
            context.append(result)
            document_id.append(examples['document_id'][i])
            ids.append(examples['id'][i])
            question.append(examples['question'][i])
            title.append(examples['title'][i])

        return {'answers': answers,
                'context': context,
                'document_id': document_id,
                'id': ids,
                'question': question,
                'title': title}
