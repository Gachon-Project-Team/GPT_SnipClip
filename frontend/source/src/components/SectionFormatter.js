import React from 'react';
import ScriptEditor from './ScriptEditor';
import ImageSelector from './ImageSelector';
import { FormatterContainer } from './Styles/Container/SectionFormatter.styles';

function SectionFormatter({
  index,
  scriptValue,
  onScriptChange,
  imageType,
  onImageTypeChange,
  realImageUrl,
  aiImageUrl
}) {
  return (
    <FormatterContainer>
      <ScriptEditor value={scriptValue} onChange={onScriptChange} />
      <ImageSelector
        realImageUrl={realImageUrl}
        aiImageUrl={aiImageUrl}
        selected={imageType}
        onChange={onImageTypeChange}
        groupName={`imageType-${index}`}
      />
    </FormatterContainer>
  );
}

export default SectionFormatter;