import styled from 'styled-components';

export const SelectorWrapper = styled.div`
  display: flex;
  flex: 4;
  justify-content: center;
  align-items: flex-start;
  gap: 1rem;
  max-width: 100%;      
  box-sizing: border-box;
`;


export const ImageBlock = styled.div`
  text-align: center;
`;

export const ClickableImage = styled.img`
  width: 100%;
  height: 150px;
  cursor: pointer;
  border-radius: 6px;
`;

export const RadioWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
`;

export const RadioLabel = styled.label`
  color: black;
  margin: 0;
`;

export const ModalImage = styled.img`
  width: 100%;
  height: auto;
`;