from papertranslator.refinement import attach_context, edit
from papertranslator.quality import make_prompt
from papertranslator.terminology import build, for_source


def ir(*texts):
    return dict(blocks=[dict(id=str(i),source=text,preserve=False,kind='paragraph',source_page_index=i) for i,text in enumerate(texts)])


def test_literal_document_definition_guides_distant_short_abbreviation():
    document=ir('Continuing professional development (CPD) improves practice.',
                'A separate paragraph.', 'CPD')
    attach_context(document,'Synthetic study',document_terms=True)
    assert ('CPD','지속적 전문성 개발(CPD)') in document['blocks'][2]['context']['terminology']
    prompt=make_prompt('CPD',{},'general','qwen',document['blocks'][2]['context'])
    assert 'CPD translates to 지속적 전문성 개발(CPD)' in prompt
    assert 'improves practice' not in prompt  # no surrounding content for short labels
    assert document['terminology']['human_approved'] is False


def test_unknown_ambiguous_and_nonliteral_definitions_are_never_invented():
    document=ir('Professional unknown phrase (ABC).', 'CPD', 'Systematic review (XYZ).')
    glossary=build(document,'general')
    assert glossary['aliases']==[]
    assert for_source('CPD and ABC',glossary)==[]
    assert for_source('systematic review',glossary)==[('systematic review','체계적 문헌고찰')]
    conflict=build(ir('Continuing professional development (CPD).','Case planning document (CPD).'),'general')
    assert conflict['aliases']==[]
    contextual=build(ir('Drawing with charcoal in art therapy.','Heating a furnace with charcoal.'),'art_therapy')
    assert for_source('Heating a furnace with charcoal.',contextual)==[]


def test_old_jobs_keep_original_context_and_targets_are_not_learning_inputs():
    document=ir('CPD','A long sentence ' * 12)
    document['blocks'][0]['target']='An unchecked model expansion.'
    attach_context(document,'Synthetic')
    assert 'terminology' not in document and document['blocks'][0]['context']=={}
    attach_context(document,'Synthetic',document_terms=True)
    assert 'terminology' not in document['blocks'][0]['context']


def test_editor_receives_the_same_terms_as_primary():
    import json
    document=ir('Continuing professional development (CPD).','CPD')
    attach_context(document,'Synthetic',document_terms=True)
    class Backend:
        def translate(self,prompt,**kwargs):
            value=json.loads(prompt.split('\n')[-1])
            assert ['CPD','지속적 전문성 개발(CPD)'] in value['TERMINOLOGY']
            return dict(text=json.dumps(dict(decision='keep',source_evidence='',draft_evidence='',explanation_ko='',revision='')))
    edit(Backend(),'CPD','지속적 전문성 개발(CPD)',context=document['blocks'][1]['context'])
