import { useEffect, useState } from 'react';
import Button from 'react-bootstrap/Button';

function SubmitButton() {
  const [isLoading, setLoading] = useState(false);

  useEffect(() => {
    function simulateNetworkRequest() {
      return new Promise(resolve => {
        setTimeout(resolve, 2000);
      });
    }

    if (isLoading) {
      simulateNetworkRequest().then(() => {
        setLoading(false);
      });
    }
  }, [isLoading]);

  const handleClick = () => setLoading(true);

  return (
    <Button
      variant="outline-dark"
      disabled={isLoading}
      onClick={!isLoading ? handleClick : null}
    >
      {isLoading ? '동영상 생성 중' : '만들기'}
    </Button>
  );
}

export default SubmitButton;