// src/components/ProgressBar.js
import React, { useEffect, useState } from 'react';
import { ProgressBarContainer } from './Styles/Container/KeywordInput.style';
import ProgressBar from 'react-bootstrap/ProgressBar';

const ProgressBarComponent = ({ onComplete }) => {
  const [progress, setProgress] = useState(0);
  
  useEffect(() => {
    // 진행바 업데이트를 0.5초마다 진행
    let progressInterval = setInterval(() => {
      setProgress((prevProgress) => {
        if (prevProgress >= 100) {
          clearInterval(progressInterval);
          onComplete(); // 완료 시 부모 컴포넌트에 알리기
          return 100;
        }
        return prevProgress + 10;
      });
    }, 500);  // 0.5초마다 진행바 업데이트

    return () => clearInterval(progressInterval); // 컴포넌트 unmount 시 clearInterval
  }, [onComplete]);

  return (
    <ProgressBarContainer>
      <ProgressBar animated now={progress} />
    </ProgressBarContainer>
  );
};

export default ProgressBarComponent;