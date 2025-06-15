// src/components/ProgressBarComponent.js
import React from 'react';
import { ProgressBarContainer } from './Styles/Container/KeywordInput.style';
import ProgressBar from 'react-bootstrap/ProgressBar';

const ProgressBarComponent = ({ progress }) => {
  return (
    <ProgressBarContainer>
      <ProgressBar animated now={progress} />
    </ProgressBarContainer>
  );
};

export default ProgressBarComponent;
