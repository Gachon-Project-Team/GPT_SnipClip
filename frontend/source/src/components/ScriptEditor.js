import React, { useState } from 'react';
import { Container, TextField, SmallButton, CharCounter } from './Styles/Container/ScriptEditor.styles';

function ScriptEditor({ value, onChange }) {
  const [isEditable, setIsEditable] = useState(false);
  const [initialLength] = useState(value.length); // 고정된 초기 길이
  const maxLength = initialLength + 10;

  const toggleEdit = () => {
    setIsEditable((prev) => !prev);
  };

  return (
    <Container>
      <TextField
        value={value}
        onChange={(e) => onChange(e.target.value)}
        readOnly={!isEditable}
        editable={isEditable}
        rows={4}
        maxLength={maxLength}
      />
      <CharCounter>{value.length} / {maxLength}</CharCounter>
      <SmallButton onClick={toggleEdit}>
        {isEditable ? '결정' : '편집'}
      </SmallButton>
    </Container>
  );
}

export default ScriptEditor;