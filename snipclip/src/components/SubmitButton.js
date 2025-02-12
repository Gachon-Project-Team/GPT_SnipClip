import { useState } from 'react';
import Button from 'react-bootstrap/Button';

function SubmitButton({ onSubmit }) {
  const [isLoading, setLoading] = useState(false);
  
  return (
    <Button
      variant="outline-dark"
      disabled={isLoading}
      onClick={!isLoading ? () => onSubmit(setLoading) : null}
      style={{ width: "20%" }}
    >
      {isLoading ? '동영상 생성 중' : '만들기'}
    </Button>
  );
  
}
export default SubmitButton;